"""
Problem 3: Naive Bayes Classifiers

We use a small binary-feature email dataset stored in a CSV file.
You will implement Naive Bayes parameter estimation and prediction.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import sys


def load_email_data(csv_path: str = "data/email_data.csv") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the email dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing columns x1, x2, x3, x4, x5, y.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_samples, 5).
    y : np.ndarray
        Label vector of shape (n_samples,), with values +1 or -1.
    """
    df = pd.read_csv(csv_path)
    X = df[["x1", "x2", "x3", "x4", "x5"]].to_numpy()
    y = df["y"].to_numpy()
    return X, y


def estimate_nb_params(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Estimate Naive Bayes parameters from data.

    Parameters
    ----------
    X : np.ndarray
        Binary feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Labels of shape (n_samples,), values in {+1, -1}.

    Returns
    -------
    params : dict
        A dictionary containing:
          - 'class_prior': dict mapping y_value -> p(y)
          - 'feature_conditional': dict mapping
            y_value -> np.ndarray of shape (n_features,)
            with p(x_i=1 | y) for each feature i.
    """
    # TODO: replace with your implementation
    raise NotImplementedError


def predict_nb(x: np.ndarray, params: Dict[str, Any]) -> int:
    """
    Predict the class for a new feature vector using Naive Bayes.

    In case of a tie in posterior probabilities, predict +1.

    Parameters
    ----------
    x : np.ndarray
        1D array of shape (n_features,) with binary features.
    params : dict
        Parameters as returned by estimate_nb_params.

    Returns
    -------
    y_pred : int
        Predicted label (+1 or -1).
    """
    # TODO: replace with your implementation
    raise NotImplementedError


def posterior_nb(x: np.ndarray, params: Dict[str, Any]) -> float:
    """
    Compute the posterior probability p(y=+1 | x) under the Naive Bayes model.

    Parameters
    ----------
    x : np.ndarray
        1D array of shape (n_features,) with binary features.
    params : dict
        Parameters as returned by estimate_nb_params.

    Returns
    -------
    p_pos : float
        Posterior probability that y = +1 given x.
    """
    # TODO: replace with your implementation
    raise NotImplementedError


def drop_feature_and_retrain(X: np.ndarray, y: np.ndarray):
    """
    Remove the first feature (x1), retrain the Naive Bayes model,
    and return predictions on the training set both before and after
    dropping x1.

    Parameters
    ----------
    X : np.ndarray
        Original feature matrix (n_samples, n_features).
    y : np.ndarray
        Labels (n_samples,).

    Returns
    -------
    preds_full : np.ndarray
        Predictions on X using all features.
    preds_reduced : np.ndarray
        Predictions on X_reduced (dropping first feature).
    """
    # TODO: replace with your implementation
    raise NotImplementedError


def compare_accuracy(csv_path: str = "data/email_data.csv") -> str:
    """
    Load the dataset, train models with and without x1,
    and compare training accuracy.

    Returns a short string:
      - "improves"
      - "degrades"
      - "stays the same"

    indicating what happens to accuracy after removing x1.

    Parameters
    ----------
    csv_path : str
        Path to the email dataset CSV.

    Returns
    -------
    verdict : str
        One of {"improves", "degrades", "stays the same"}.
    """
    # TODO: replace with your implementation
    raise NotImplementedError


def main() -> int:
    """
    Minimal demo runner for Problem 3.
    """

    # load_email_data
    try:
        X, y = load_email_data()
        print(f"load_email_data: X.shape={X.shape}, y.shape={y.shape}")
    except Exception as e:
        print(f"load_email_data: error: {e}")
        return 1

    # estimate_nb_params
    try:
        params = estimate_nb_params(X, y)
        print("estimate_nb_params: OK")
    except NotImplementedError:
        print("estimate_nb_params: NotImplemented")
        return 0
    except Exception as e:
        print(f"estimate_nb_params: error: {e}")
        return 1

    # predict_nb on first example
    try:
        y_pred0 = predict_nb(X[0], params)
        print(f"predict_nb: first example prediction={y_pred0}")
    except NotImplementedError:
        print("predict_nb: NotImplemented")
    except Exception as e:
        print(f"predict_nb: error: {e}")

    # posterior_nb on first example
    try:
        p_pos0 = posterior_nb(X[0], params)
        print(f"posterior_nb: p(y=+1|x0)={p_pos0:.4f}")
    except NotImplementedError:
        print("posterior_nb: NotImplemented")
    except Exception as e:
        print(f"posterior_nb: error: {e}")

    # drop_feature_and_retrain
    try:
        preds_full, preds_reduced = drop_feature_and_retrain(X, y)
        print(
            f"drop_feature_and_retrain: preds_full.shape={getattr(preds_full, 'shape', None)}, "
            f"preds_reduced.shape={getattr(preds_reduced, 'shape', None)}"
        )
    except NotImplementedError:
        print("drop_feature_and_retrain: NotImplemented")
    except Exception as e:
        print(f"drop_feature_and_retrain: error: {e}")

    # compare_accuracy
    try:
        verdict = compare_accuracy()
        print(f"compare_accuracy: {verdict}")
    except NotImplementedError:
        print("compare_accuracy: NotImplemented")
    except Exception as e:
        print(f"compare_accuracy: error: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
