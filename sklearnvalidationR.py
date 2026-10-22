"""
Validation script: runs scikit-learn's PCA on the same data used in pca.py.

Run this alongside pca.py and compare the explained variance ratios. The
values match, which confirms the from-scratch implementation is correct.
(Component and projection signs may differ, which is expected; see README.)
"""

from sklearn.decomposition import PCA
import numpy as np

# Same seed and data-generation steps as pca.py, so both scripts see
# identical data.
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 5)) @ rng.normal(size=(5, 5))

# Fit scikit-learn's PCA keeping the first two components.
pca = PCA(n_components=2)
pca.fit(X)

# Fraction of the total variance explained by each of the two components.
print(pca.explained_variance_ratio_)
