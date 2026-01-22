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
    try:
        import torchvision
        import torchvision.transforms as transforms
    except ImportError:
        raise ImportError("torchvision is required. Install with: pip install torchvision")
    
    # Load training data
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True,
        transform=transforms.ToTensor()
    )
    
    # Load test data
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=True,
        transform=transforms.ToTensor()
    )
    
    # Convert to numpy arrays
    X_train = []
    y_train = []
    for img, label in train_dataset:
        # img is already normalized to [0,1] by ToTensor()
        X_train.append(img.numpy().flatten())
        y_train.append(label)
    
    X_test = []
    y_test = []
    for img, label in test_dataset:
        X_test.append(img.numpy().flatten())
        y_test.append(label)
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)
    
    return X_train, y_train, X_test, y_test


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
    n_test = X_test.shape[0]
    n_train = X_train.shape[0]
    
    # Using vectorized computation for efficiency
    # Euclidean distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    
    # Compute squared norms
    test_sq = np.sum(X_test**2, axis=1, keepdims=True)  # (n_test, 1)
    train_sq = np.sum(X_train**2, axis=1, keepdims=True).T  # (1, n_train)
    
    # Compute dot product
    dot_product = X_test @ X_train.T  # (n_test, n_train)
    
    # Compute squared distances
    dists_sq = test_sq + train_sq - 2 * dot_product
    
    # Take square root (ensure non-negative due to floating point errors)
    dists = np.sqrt(np.maximum(dists_sq, 0))
    
    return dists


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
    n_test = dists.shape[0]
    y_pred = np.zeros(n_test, dtype=y_train.dtype)
    
    for i in range(n_test):
        # Get indices of k nearest neighbors
        k_nearest_indices = np.argsort(dists[i])[:k]
        
        # Get labels of k nearest neighbors
        k_nearest_labels = y_train[k_nearest_indices]
        
        # Find most common label (mode)
        # Using bincount for efficiency
        y_pred[i] = np.bincount(k_nearest_labels).argmax()
    
    return y_pred


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
    # Compute distances once
    dists = compute_distances(X_train, X_test)
    
    accuracies = {}
    for k in k_values:
        # Predict with this k
        y_pred = predict_knn(dists, y_train, k)
        
        # Compute accuracy
        accuracy = np.mean(y_pred == y_test)
        accuracies[k] = float(accuracy)
    
    return accuracies


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