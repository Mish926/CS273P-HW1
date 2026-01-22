"""
Problem 1: Python & Data Exploration

Functions in this file operate on the Iris dataset.
You are expected to implement each function so that it returns
the specified values. The autograder will import and call these
functions directly.
"""

from typing import Tuple, List
import numpy as np
import sys


def get_shape(X: np.ndarray) -> Tuple[int, int]:
    """
    Return the number of data points and the number of features.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (n_samples, n_features).

    Returns
    -------
    n_samples : int
    n_features : int
    """
    # TODO: replace with your implementation
    raise NotImplementedError


def feature_histograms(X: np.ndarray, bins: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute histogram counts for each feature (column) in X.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (n_samples, n_features).
    bins : int
        Number of histogram bins.

    Returns
    -------
    histograms : list of (hist, bin_edges)
        A list of length n_features, where each element is a tuple:
        (hist, bin_edges) as returned by np.histogram for that feature.
    """
    # TODO: replace with your implementation
    raise NotImplementedError


def compute_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean and standard deviation for each feature.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (n_samples, n_features).

    Returns
    -------
    means : np.ndarray
        1D array of shape (n_features,) with feature means.
    stds : np.ndarray
        1D array of shape (n_features,) with feature standard deviations.
    """
    # TODO: replace with your implementation
    raise NotImplementedError


def scatter_pairs(
    X: np.ndarray,
    y: np.ndarray,
    feature_pairs: Tuple[Tuple[int, int], ...] = ((0, 1), (0, 2), (0, 3)),
):
    """
    Prepare coordinate pairs for selected feature index pairs, grouped by class.

    This function does NOT plot. It only returns the values that would be used
    to make scatter plots.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (n_samples, n_features).
    y : np.ndarray
        1D array of shape (n_samples,) with class labels.
    feature_pairs : tuple of (int, int)
        Feature index pairs to consider (0-based indexing).

    Returns
    -------
    result : dict
        A dictionary mapping (i, j) -> {class_label: points},
        where points is an array of shape (n_points_in_class, 2)
        containing the selected feature values.
    """
    # TODO: replace with your implementation
    raise NotImplementedError


def _load_iris() -> Tuple[np.ndarray, np.ndarray]:
    """
    Best-effort loader for the Iris dataset.

    - Loads via scikit-learn and returns data and labels.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    """
    try:
        from sklearn.datasets import load_iris  # type: ignore
    except Exception as e:
        raise ImportError(
            "scikit-learn is required to load the Iris dataset. Install with: pip install scikit-learn"
        ) from e

    data = load_iris()
    X = data.data.astype(np.float64)
    y = data.target.astype(np.int64)
    return X, y


def main(argv: List[str] = None) -> int:
    """
    Minimal runner to call the assignment functions.
    """

    X, y = _load_iris()

    # get_shape
    try:
        n_samples, n_features = get_shape(X)
        print(f"get_shape: n_samples={n_samples}, n_features={n_features}")
    except NotImplementedError:
        print("get_shape: NotImplemented")

    # feature_histograms
    try:
        n_bins = 10
        hists = feature_histograms(X, bins=n_bins)
        print(f"feature_histograms: computed {len(hists)} histograms with bins={n_bins}")
    except NotImplementedError:
        print("feature_histograms: NotImplemented")

    # compute_stats
    try:
        means, stds = compute_stats(X)
        print(f"compute_stats: means.shape={means.shape}, stds.shape={stds.shape}")
    except NotImplementedError:
        print("compute_stats: NotImplemented")

    # scatter_pairs
    try:
        result = scatter_pairs(X, y)
        print(f"scatter_pairs: prepared {len(result)} feature pair groups")
    except NotImplementedError:
        print("scatter_pairs: NotImplemented")

    return 0


if __name__ == "__main__":
    sys.exit(main())
