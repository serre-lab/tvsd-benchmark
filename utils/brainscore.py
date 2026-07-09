import numpy as np
from time import time
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def spearman_brown(r, n: int = 2):
    """Spearman-Brown correction.

    Extrapolates the reliability of a measurement composed of `n` equal parts
    from the correlation `r` between single parts. With `n=2` (split-half) this
    is the `2r / (1 + r)` form used by Brain-Score's InternalConsistency ceiling.
    """
    return n * r / (1 + (n - 1) * r)


def brain_score_pearsonr(Y_pred, Y_test):
    """
    Compute the Pearson's correlation between the predicted and actual labels.

    Parameters
    ----------
    x : np.ndarray
        Predicted labels, shape (aggregated_spatial_dim, num_features).
    y : np.ndarray
        Actual labels, shape (aggregated_spatial_dim, num_features).

    Returns
    -------
    r : np.ndarray
        Pearson correlation coefficients for each feature, shape (num_features,).
    """

    # Compute the Spearman correlation for each neuron
    pearsonr_correlations = np.zeros(Y_test.shape[-1])
    for i in range(Y_test.shape[-1]):
        corr, _ = pearsonr(Y_test[:, i], Y_pred[:, i])
        pearsonr_correlations[i] = corr

    return pearsonr_correlations


def brain_score_spearman(Y_pred, Y_test):
    """
    Compute the Spearman's rank correlation between the predicted and actual labels for each neuron,
    averaged across all spatial positions and images.

    Parameters
    ----------
    Y_pred : np.ndarray
        Predicted labels, shape (bs * h * w, num_features).
    Y_test : np.ndarray
        Actual labels, shape (bs * h * w, num_features).

    Returns
    -------
    spearman_scores : np.ndarray
        Mean Spearman correlation coefficients for each feature, shape (num_features,).
    """

    # Compute the Spearman correlation for each neuron
    spearman_correlations = np.zeros(Y_test.shape[-1])
    for i in range(Y_test.shape[-1]):
        corr, _ = spearmanr(Y_test[:, i], Y_pred[:, i])
        spearman_correlations[i] = corr

    # print('brain_score_spearman: ', spearman_correlations)
    return spearman_correlations


def _make_splitter(cv_strategy, n_splits, train_size, random_state):
    """Build the cross-validation splitter.

    Brain-Score's default is a ShuffleSplit with 10 splits and train_size=0.9,
    which we mirror here. KFold is kept as an option for non-overlapping folds.
    """
    if cv_strategy == "shuffle":
        return ShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=random_state)
    elif cv_strategy == "kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        raise ValueError(f"Unknown cv_strategy: {cv_strategy}")


def _fit_predict(X_train, Y_train, X_test, pca_components=100, skip_pca=False, standardize=False):
    """Fit the (optional standardize -> optional PCA -> PLS) pipeline on the
    training split and predict the test split.

    The scaler and PCA are fit on ``X_train``/``Y_train`` only and applied to the
    test split, so there is no train->test leakage in the mapping. Returns the
    predictions on the test split, shape (n_test_stimuli, n_neuroids).
    """
    if standardize:
        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)
        Y_train = StandardScaler().fit_transform(Y_train)

    if not skip_pca and pca_components is not None and X_train.shape[-1] > pca_components:
        pca_X = PCA(n_components=min(X_train.shape[-1], pca_components))
        X_train = pca_X.fit_transform(X_train)
        X_test = pca_X.transform(X_test)

    n_components = min(X_train.shape[-1], Y_train.shape[-1], 25)
    pls_reg = PLSRegression(n_components=n_components, scale=False)
    pls_reg.fit(X_train, Y_train)
    return pls_reg.predict(X_test)


def _correlations(Y_pred, Y_test, correlation_fn="pearson"):
    """Per-neuroid correlation between predicted and actual responses."""
    if correlation_fn == "pearson":
        return brain_score_pearsonr(Y_pred, Y_test)
    elif correlation_fn == "spearman":
        return brain_score_spearman(Y_pred, Y_test)
    else:
        raise ValueError("Unknown correlation metric")


def _reduce(correlations, reducer="median"):
    """Aggregate per-neuroid correlations to a single score."""
    if reducer == "median":
        return np.nanmedian(correlations)
    elif reducer == "mean":
        return np.nanmean(correlations)
    else:
        raise ValueError("Unknown reducer")


def score_train_test(
    X_train,
    Y_train,
    X_test,
    Y_test,
    reducer="median",
    correlation_fn="pearson",
    pca_components=100,
    skip_pca=False,
    standardize=False,
    ceiling=None,
    ceiling_normalize=False,
    n_boot_ci=100,
    random_state=42,
):
    """Neural predictivity on a held-out test split (the principled TVSD path).

    Fit the mapping on the train split, predict the test split, correlate per
    neuroid across the test stimuli, and reduce across neuroids. The train/test
    split *is* the evaluation, so there is no cross-validation; the score's spread
    is instead estimated by bootstrapping over the test stimuli.

    Args:
        X_train, Y_train: model features / neural responses on the train split.
        X_test, Y_test: model features / neural responses on the held-out test
            split (for TVSD, Y_test is the repetition-averaged ``test_MUA``).
        ceiling: per-neuroid noise-ceiling vector aligned to the Y columns
            (for TVSD, the split-half + Spearman-Brown internal consistency of the
            repetition-averaged test responses).
        ceiling_normalize: if True, divide the score by the median ceiling so that
            1.0 means "predicts as well as the noise ceiling allows".
        n_boot_ci: number of bootstrap resamples over test stimuli for the std.

    Returns:
        (layer_score, layer_std).
    """
    Y_pred = _fit_predict(X_train, Y_train, X_test, pca_components, skip_pca, standardize)

    def _scalar(pred, actual):
        s = _reduce(_correlations(pred, actual, correlation_fn), reducer)
        if ceiling_normalize:
            if ceiling is None:
                raise ValueError("ceiling_normalize=True requires a ceiling vector")
            s = s / np.nanmedian(ceiling)
        return s

    layer_score = _scalar(Y_pred, Y_test)

    # Spread via bootstrap over the test stimuli (rows), reusing the fixed
    # predictions -- the mapping is not refit.
    rng = np.random.default_rng(random_state)
    n_test = Y_test.shape[0]
    boot = [
        _scalar(Y_pred[idx], Y_test[idx])
        for idx in (rng.integers(0, n_test, n_test) for _ in range(n_boot_ci))
    ]
    layer_std = float(np.nanstd(boot))

    print(f"Layer score: {layer_score:.4f}, Layer std: {layer_std:.4f}")
    return float(layer_score), layer_std


def compute_brain_score(
    X,
    Y,
    n_splits=10,
    train_size=0.9,
    cv_strategy="shuffle",
    reducer="median",
    correlation_fn="pearson",
    pca_components=100,
    skip_pca=False,
    standardize=False,
    ceiling=None,
    ceiling_normalize=False,
    random_state=42,
):
    """Cross-validated neural predictivity within a single assembly (legacy path).

    Kept for comparison; the TVSD pipeline now prefers ``score_train_test`` (fit on
    the train split, score the held-out test split). For each CV split, fit the
    mapping on the train fold and correlate predictions on the test fold, then
    average scores across folds.
    """
    splitter = _make_splitter(cv_strategy, n_splits, train_size, random_state)
    scores = []
    times = []
    for train_index, test_index in splitter.split(X):
        start_time = time()
        Y_pred = _fit_predict(
            X[train_index],
            Y[train_index],
            X[test_index],
            pca_components,
            skip_pca,
            standardize,
        )
        correlations = _correlations(Y_pred, Y[test_index], correlation_fn)
        scores.append(_reduce(correlations, reducer))
        times.append(time() - start_time)

    layer_score = np.nanmean(scores)
    layer_std = np.nanstd(scores)

    if ceiling_normalize:
        if ceiling is None:
            raise ValueError("ceiling_normalize=True requires a ceiling vector")
        ceiling_center = np.nanmedian(ceiling)
        layer_score = layer_score / ceiling_center
        layer_std = layer_std / ceiling_center

    print(
        f"Layer score: {layer_score:.4f}, Layer std: {layer_std:.4f}, "
        f"Mean time per fold: {np.mean(times):.4f} seconds"
    )
    return layer_score, layer_std
