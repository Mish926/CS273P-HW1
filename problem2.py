"""
Problem 2: kNN Predictions on CIFAR-10

Implement a simple k-nearest neighbors classifier.
The autograder will import and call these functions.
"""

from typing import Tuple, Dict
import numpy as np
import sys


def load_cifar10(data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CIFAR-10 data using torchvision and return as numpy arrays.

    Parameters
    ----------
    data_dir : str
        Directory where CIFAR-10 data is stored.

    Returns
    -------
    X_train : np.ndarray
    y_train : np.ndarray
    X_test : np.ndarray
    y_test : np.ndarray
    """
    # TODO: Implement using torchvision.datasets.CIFAR10.
    # Return numpy arrays (X_train, y_train, X_test, y_test) where
    # images are normalized to [0,1] and flattened to shape (n, d),
    # and labels are dtype np.int64.
    raise NotImplementedError("Implement using torchvision.datasets.CIFAR10 and return numpy arrays")


def compute_distances(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distances between training and test samples.

    Parameters
    ----------
    X_train : np.ndarray
        Training data of shape (n_train, d).
    X_test : np.ndarray
        Test data of shape (n_test, d).

    Returns
    -------
    dists : np.ndarray
        Distance matrix of shape (n_test, n_train),
        where dists[i, j] is the distance between X_test[i] and X_train[j].
    """
    # TODO: replace with your implementation
    raise NotImplementedError


def predict_knn(
    dists: np.ndarray,
    y_train: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Predict labels for test data based on precomputed distances. 

    Parameters
    ----------
    dists : np.ndarray
        Distance matrix of shape (n_test, n_train).
    y_train : np.ndarray
        Training labels of shape (n_train,).
    k : int
        Number of nearest neighbors to use.

    Returns
    -------
    y_pred : np.ndarray
        Predicted labels of shape (n_test,).
    """
    # TODO: replace with your implementation
    raise NotImplementedError


def evaluate_accuracy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    k_values: Tuple[int, ...] = (1, 3, 5),
) -> Dict[int, float]:
    """
    Evaluate kNN accuracy for several k values.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : np.ndarray
        Train/test data and labels.
    k_values : tuple of int
        Set of k values to evaluate.

    Returns
    -------
    accuracies : dict
        A dictionary mapping k -> accuracy (0.0 to 1.0).
    """
    # TODO: replace with your implementation
    raise NotImplementedError


def main() -> int:
    """
    Minimal demo runner for Problem 2.
    """

    # load_cifar10
    try:
        X_train, y_train, X_test, y_test = load_cifar10("data")
        print(
            f"load_cifar10: X_train={X_train.shape}, y_train={y_train.shape}, "
            f"X_test={X_test.shape}, y_test={y_test.shape}"
        )
    except NotImplementedError:
        print("load_cifar10: NotImplemented")
        return 0
    except Exception as e:
        print(f"load_cifar10: error: {e}")
        return 1

    # compute_distances
    try:
        dists = compute_distances(X_train, X_test)
        print(f"compute_distances: dists.shape={dists.shape}")
    except NotImplementedError:
        print("compute_distances: NotImplemented")
        return 0
    except Exception as e:
        print(f"compute_distances: error: {e}")
        return 1

    # predict_knn for a few k values
    for k in (1, 3, 5):
        try:
            y_pred = predict_knn(dists, y_train, k)
            acc = float(np.mean(y_pred == y_test))
            print(f"predict_knn: k={k}, accuracy={acc:.4f}")
        except NotImplementedError:
            print(f"predict_knn: k={k} NotImplemented")
        except Exception as e:
            print(f"predict_knn: k={k} error: {e}")

    # evaluate_accuracy
    try:
        accuracies = evaluate_accuracy(X_train, y_train, X_test, y_test, (1, 3, 5))
        print(f"evaluate_accuracy: {accuracies}")
    except NotImplementedError:
        print("evaluate_accuracy: NotImplemented")
    except Exception as e:
        print(f"evaluate_accuracy: error: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
