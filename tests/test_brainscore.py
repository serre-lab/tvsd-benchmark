"""Tests for utils.brainscore scoring helpers."""

import numpy as np
import pytest

from utils.brainscore import (
    compute_brain_score,
    score_train_test,
    spearman_brown,
    _make_splitter,
)


class TestSpearmanBrown:
    def test_known_values(self):
        # 2r/(1+r)
        assert spearman_brown(0.5) == pytest.approx(2 * 0.5 / 1.5)
        assert spearman_brown(0.0) == pytest.approx(0.0)
        assert spearman_brown(1.0) == pytest.approx(1.0)

    def test_inflates_positive_correlation(self):
        # For r in (0, 1), the SB-corrected value exceeds the raw half-split r.
        for r in [0.1, 0.3, 0.6, 0.9]:
            assert spearman_brown(r) > r

    def test_n_parts_generalization(self):
        # n=1 is the identity.
        assert spearman_brown(0.4, n=1) == pytest.approx(0.4)


class TestMakeSplitter:
    def test_shuffle(self):
        from sklearn.model_selection import ShuffleSplit

        sp = _make_splitter("shuffle", n_splits=10, train_size=0.9, random_state=0)
        assert isinstance(sp, ShuffleSplit)
        assert sp.get_n_splits() == 10

    def test_kfold(self):
        from sklearn.model_selection import KFold

        sp = _make_splitter("kfold", n_splits=5, train_size=0.9, random_state=0)
        assert isinstance(sp, KFold)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown cv_strategy"):
            _make_splitter("bogus", n_splits=5, train_size=0.9, random_state=0)


class TestComputeBrainScore:
    def _linear_data(self):
        # Y is a linear function of the first few features -> PLS predicts well.
        rng = np.random.default_rng(0)
        X = rng.standard_normal((120, 20))
        W = rng.standard_normal((20, 5))
        Y = X @ W + rng.standard_normal((120, 5)) * 0.05
        return X, Y

    def test_predicts_correlated_data(self):
        X, Y = self._linear_data()
        score, std = compute_brain_score(X, Y)
        # Strong linear relationship -> high positive predictivity.
        assert score > 0.8
        assert std >= 0

    def test_ceiling_normalization_scales_score(self):
        X, Y = self._linear_data()
        raw, _ = compute_brain_score(X, Y)
        ceiling = np.full(Y.shape[1], 0.5)
        normed, _ = compute_brain_score(X, Y, ceiling=ceiling, ceiling_normalize=True)
        # Dividing by a median ceiling of 0.5 doubles the score.
        assert normed == pytest.approx(raw / 0.5, rel=1e-6)

    def test_ceiling_normalize_requires_ceiling(self):
        X, Y = self._linear_data()
        with pytest.raises(ValueError, match="requires a ceiling"):
            compute_brain_score(X, Y, ceiling_normalize=True)

    def test_reproducible(self):
        X, Y = self._linear_data()
        s1, _ = compute_brain_score(X, Y, random_state=7)
        s2, _ = compute_brain_score(X, Y, random_state=7)
        assert s1 == pytest.approx(s2)


class TestScoreTrainTest:
    def _split_data(self):
        # Train and test share one linear map -> fit-on-train predicts test well.
        rng = np.random.default_rng(0)
        W = rng.standard_normal((20, 5))
        X_train = rng.standard_normal((500, 20))
        Y_train = X_train @ W + rng.standard_normal((500, 5)) * 0.05
        X_test = rng.standard_normal((100, 20))
        Y_test = X_test @ W + rng.standard_normal((100, 5)) * 0.05
        return X_train, Y_train, X_test, Y_test

    def test_predicts_heldout_test(self):
        X_tr, Y_tr, X_te, Y_te = self._split_data()
        score, std = score_train_test(X_tr, Y_tr, X_te, Y_te)
        assert score > 0.8
        assert std >= 0

    def test_ceiling_normalization_scales_score(self):
        X_tr, Y_tr, X_te, Y_te = self._split_data()
        raw, _ = score_train_test(X_tr, Y_tr, X_te, Y_te)
        ceiling = np.full(Y_te.shape[1], 0.5)
        normed, _ = score_train_test(
            X_tr, Y_tr, X_te, Y_te, ceiling=ceiling, ceiling_normalize=True
        )
        assert normed == pytest.approx(raw / 0.5, rel=1e-6)

    def test_ceiling_normalize_requires_ceiling(self):
        X_tr, Y_tr, X_te, Y_te = self._split_data()
        with pytest.raises(ValueError, match="requires a ceiling"):
            score_train_test(X_tr, Y_tr, X_te, Y_te, ceiling_normalize=True)

    def test_std_reproducible_and_nonzero(self):
        X_tr, Y_tr, X_te, Y_te = self._split_data()
        s1, std1 = score_train_test(X_tr, Y_tr, X_te, Y_te, random_state=7)
        s2, std2 = score_train_test(X_tr, Y_tr, X_te, Y_te, random_state=7)
        assert s1 == pytest.approx(s2)
        assert std1 == pytest.approx(std2)  # deterministic bootstrap
        assert std1 > 0  # bootstrap over test stimuli gives real spread

    def test_noise_features_score_near_zero(self):
        # Features unrelated to responses -> predictivity near 0 on held-out test.
        rng = np.random.default_rng(1)
        X_tr = rng.standard_normal((500, 20))
        Y_tr = rng.standard_normal((500, 5))
        X_te = rng.standard_normal((100, 20))
        Y_te = rng.standard_normal((100, 5))
        score, _ = score_train_test(X_tr, Y_tr, X_te, Y_te)
        assert abs(score) < 0.2
