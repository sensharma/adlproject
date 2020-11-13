from sklearn.preprocessing import MinMaxScaler
import numpy as np


def scale_zero_one(X):
    """
    Scale inputs between 0 and 1 as per SHAP paper
    Should work with Conv2d for both keras (N, h, w, c) and PyTorch (N, c, h, w)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler takes data shape (#samples x #features, so reshaping ->)
    X_res = X.reshape(X.shape[0], -1)
    X_scaled = scaler.fit_transform(X_res)
    return X_scaled.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3])


def viz_scores(scores, ax):
    reshaped_scores = scores.reshape(28, 28)
    the_min = np.min(reshaped_scores)
    the_max = np.max(reshaped_scores)
    center = 0.0
    negative_vals = (reshaped_scores < center) * \
        reshaped_scores/(the_min + 10**-7)
    positive_vals = (reshaped_scores > center)*reshaped_scores/float(the_max)
    reshaped_scores = -negative_vals + positive_vals
    # ax.imshow(-reshaped_scores, cmap="Greys") # original
    ax.imshow(-reshaped_scores, cmap='viridis')
    ax.set_xticks([])
    ax.set_yticks([])
