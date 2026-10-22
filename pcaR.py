"""
Principal Component Analysis (PCA) implemented from scratch using NumPy.

The result has been validated against scikit-learn's
`sklearn.decomposition.PCA` (see sklearnvalidation.py).
"""

import numpy as np

# Optional: load the Iris dataset to try the model on real data.
# import kagglehub
#
# # Download latest version
# path = kagglehub.dataset_download("uciml/iris")
#
# print("Path to dataset files:", path)


def pca(X, n_components):
    """
    Perform PCA using eigendecomposition of the covariance matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input data.
    n_components : int
        The number of principal components to keep.

    Returns
    -------
    X_projected : ndarray of shape (n_samples, n_components)
        The data projected onto the principal components.
    components : ndarray of shape (n_features, n_components)
        The principal axes (one per column), sorted by variance explained.
    explained_variance_ratio : ndarray of shape (n_components,)
        The fraction of the total variance explained by each component.
    mean : ndarray of shape (n_features,)
        The per-feature mean, needed to reconstruct the original data.
    """
    # 1. Centre the data. PCA looks for directions of maximum variance
    #    around the mean, so every feature must have zero mean.
    mean = X.mean(axis=0)
    X_centered = X - mean

    # 2. Compute the covariance matrix of the features (n_features x n_features).
    #    rowvar=False tells NumPy that columns are variables and rows are samples.
    cov = np.cov(X_centered, rowvar=False)

    # 3. Eigendecomposition. The covariance matrix is symmetric, so `eigh` is
    #    the right (and faster, more stable) choice over `eig`. Eigenvectors are
    #    the principal directions; eigenvalues are the variance along them.
    eigvals, eigvecs = np.linalg.eigh(cov)

    # 4. `eigh` returns eigenvalues in ascending order, so sort them in
    #    descending order to put the most informative components first.
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # 5. Keep only the top `n_components` eigenvectors.
    components = eigvecs[:, :n_components]

    # 6. Project the centred data onto the selected components.
    X_projected = X_centered @ components

    # 7. Each eigenvalue is the variance along its component, so dividing by
    #    the sum of all eigenvalues (the total variance) gives the fraction of
    #    variance explained. This uses ALL eigenvalues in the denominator, not
    #    just the ones that were kept.
    explained_variance_ratio = eigvals[:n_components] / eigvals.sum()

    return X_projected, components, explained_variance_ratio, mean


# Example
# Generate correlated data: multiplying independent normal samples by a random
# matrix mixes the features, which gives PCA some structure to find.
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 5)) @ rng.normal(size=(5, 5))

X_proj, components, evr, mean = pca(X, n_components=3)
print(components, evr)  # principal axes, and fraction of variance explained by each component

# Reconstruct an approximation of the original data from the reduced
# representation (the information in the dropped components is lost).
# X_approx = X_proj @ components.T + mean
