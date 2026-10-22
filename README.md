# Principal Component Analysis (PCA) from Scratch using only Numpy.

Using NumPy, I built a Principal Component Analysis (PCA) model from scratch. The model produces the same values as scikit-learn's `sklearn.decomposition.PCA`, which I used to validate it.

## Files

| File | Description |
| --- | --- |
| `pca.py` | PCA implemented from scratch with NumPy |
| `sklearnvalidation.py` | The same analysis using scikit-learn, used to validate the results |

## Usage

```bash
pip install numpy scikit-learn
python pca.py
python sklearnvalidation.py
```

```python
X_projected, components, explained_variance_ratio, mean = pca(X, n_components=2)
```

Both scripts generate the same random dataset from a fixed seed, so their outputs can be compared directly. The explained variance ratios from the two scripts match.

> `pca.py` keeps 3 components while `sklearnvalidation.py` keeps 2. Compare the first two values, or set `n_components=2` in both for an exact side-by-side comparison.

## How it works

1. **Centre the data** by subtracting the mean of each feature.
2. **Compute the covariance matrix** of the features.
3. **Eigendecompose** the covariance matrix. The eigenvectors are the principal components and the eigenvalues are the variance along each one.
4. **Sort** the components by eigenvalue, largest first.
5. **Keep** the top `n_components` eigenvectors.
6. **Project** the centred data onto those components.

## Interesting parts of a PCA model

### Principal components are directions of maximum variance
Each principal component is a direction in feature space. The first captures the most variance in the data, and each following component captures the most remaining variance while being orthogonal (at right angles) to all the previous ones.

### Eigenvalues measure the variance explained
Each eigenvalue is the variance of the data along its eigenvector. Dividing an eigenvalue by the sum of all eigenvalues gives the **explained variance ratio**, which shows how much of the data's information a component holds. Summing the ratios of the kept components tells you how much information the reduced data retains.

### Why `eigh` instead of `eig`
A covariance matrix is symmetric, so `np.linalg.eigh` is the right choice. It is faster and more numerically stable than `eig`, and it guarantees real eigenvalues. It returns eigenvalues in ascending order, which is why the code reverses the order.

### Centring is essential
PCA finds directions of variance *around the mean*. Without centring, the first component would mostly point towards the data's mean rather than along its spread.

### Signs can differ from scikit-learn
An eigenvector and its negative describe the same direction, so a component can legitimately point either way. Your projected values may be sign-flipped compared with scikit-learn. This is expected and does not affect the result. The explained variance ratios are unaffected.

### Scale matters
PCA is sensitive to the scale of the features. A feature measured in large units dominates the variance and therefore the first component. When features use different units, standardise them (zero mean, unit variance) before applying PCA.

### Dimensionality reduction and reconstruction
Projecting onto fewer components compresses the data. You can approximate the original data from the reduced version:

```python
X_approx = X_projected @ components.T + mean
```

The more variance the kept components explain, the closer the approximation is to the original. The information in the discarded components is lost.

### Scikit-learn uses SVD
Scikit-learn computes PCA with a Singular Value Decomposition (SVD) of the centred data rather than an eigendecomposition of the covariance matrix. The two approaches are mathematically equivalent and give the same results, but SVD is generally more numerically stable and avoids forming the covariance matrix explicitly.
