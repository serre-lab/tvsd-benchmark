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
    """Cross-validated neural predictivity (Brain-Score's CrossRegressedCorrelation).

    For each CV split: (optionally standardize, optionally PCA-reduce the model
    features), fit PLS regression (n_components<=25, scale=False) from features to
    neural responses, then correlate predicted vs. actual per neuroid and reduce
    across neuroids. The per-split scores are averaged.

    Args:
        X: model activations, shape (stimuli, features).
        Y: neural responses, shape (stimuli, neuroids).
        pca_components: if not None and features exceed it, PCA-reduce. The PCA is
            fit on the training fold only (no test leakage). NOTE: if X was already
            reduced by the generate-time IncrementalPCA, that basis was fit over the
            whole image set (train+test) and therefore leaks; pass skip_pca=True and
            be aware of that caveat. Prefer per-fold PCA here for a clean estimate.
        skip_pca: skip in-fold PCA (e.g. features are already reduced).
        standardize: z-score features and targets (fit on train). Off by default to
            match Brain-Score, which relies on PLS's internal centering (scale=False).
        ceiling: optional per-neuroid noise-ceiling vector (aligned to Y columns).
        ceiling_normalize: if True and ceiling given, divide the score by the median
            ceiling (Brain-Score-style normalization: 1.0 == ceiling-level prediction).
    """
    splitter = _make_splitter(cv_strategy, n_splits, train_size, random_state)
    scores = []
    times = []
    for train_index, test_index in splitter.split(X):
        start_time = time()
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        if standardize:
            scaler_X, scaler_Y = StandardScaler(), StandardScaler()
            X_train = scaler_X.fit_transform(X_train)
            Y_train = scaler_Y.fit_transform(Y_train)
            X_test = scaler_X.transform(X_test)
            Y_test = scaler_Y.transform(Y_test)

        if not skip_pca and pca_components is not None and X_train.shape[-1] > pca_components:
            print("Performing PCA (fit on train fold)...")
            pca_X = PCA(n_components=min(X_train.shape[-1], pca_components))
            X_train = pca_X.fit_transform(X_train)
            X_test = pca_X.transform(X_test)

        n_components = min(X_train.shape[-1], Y_train.shape[-1], 25)
        print("Performing PLS regression...")
        pls_reg = PLSRegression(n_components=n_components, scale=False)
        pls_reg.fit(X_train, Y_train)
        Y_pred = pls_reg.predict(X_test)

        if correlation_fn == "pearson":
            correlations = brain_score_pearsonr(Y_pred, Y_test)
        elif correlation_fn == "spearman":
            correlations = brain_score_spearman(Y_pred, Y_test)
        else:
            raise ValueError("Unknown correlation metric")

        if reducer == "median":
            score = np.nanmedian(correlations)
        elif reducer == "mean":
            score = np.nanmean(correlations)
        else:
            raise ValueError("Unknown reducer")

        scores.append(score)
        end_time = time()
        times.append(end_time - start_time)

    layer_score = np.nanmean(scores)
    layer_std = np.nanstd(scores)

    if ceiling_normalize:
        if ceiling is None:
            raise ValueError("ceiling_normalize=True requires a ceiling vector")
        ceiling_center = np.nanmedian(ceiling)
        # Divide both the score and its spread by the (scalar) ceiling so a
        # normalized score of 1.0 means "predicts as well as the noise ceiling".
        layer_score = layer_score / ceiling_center
        layer_std = layer_std / ceiling_center

    mean_time = np.mean(times)
    print(
        f"Layer score: {layer_score:.4f}, Layer std: {layer_std:.4f}, "
        f"Mean time per fold: {mean_time:.4f} seconds"
    )

    return layer_score, layer_std
