import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torchvision.transforms.functional import resize, to_pil_image

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

def compute_brain_score(X, Y, n_splits=4, reducer='median', correlation_fn='pearson'):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        scaler_X, scaler_Y = StandardScaler(), StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        Y_train = scaler_Y.fit_transform(Y_train)
        X_test = scaler_X.transform(X_test)
        Y_test = scaler_Y.transform(Y_test)

        print("Performing PCA...")
        pca_X = PCA(n_components=min(X_train.shape[-1], 100))
        X_train = pca_X.fit_transform(X_train)
        X_test = pca_X.transform(X_test)

        n_components = min(X_train.shape[-1], Y_train.shape[-1], 25)
        print("Performing PLS regression...")
        pls_reg = PLSRegression(n_components=n_components, scale=False)
        pls_reg.fit(X_train, Y_train)
        Y_pred = pls_reg.predict(X_test)

        if correlation_fn == 'pearson':
            correlations = brain_score_pearsonr(Y_pred, Y_test) # we are interested in the trend and no need to interpret the results in the original scale
        elif correlation_fn == 'spearman':
            correlations = brain_score_spearman(Y_pred, Y_test)
        else:
            raise ValueError('Unknown correlation metric')
        
        if reducer == 'median':
            score = np.nanmedian(correlations)  
        elif reducer == 'mean':   
            score = np.nanmean(correlations)  
        else:
            raise ValueError('Unknown reducer')
        
        scores.append(score)

    layer_score = np.nanmean(scores)
    layer_std = np.nanstd(scores)
    return layer_score, layer_std
