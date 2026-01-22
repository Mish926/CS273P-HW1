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
    n_samples, n_features = X.shape
    unique_classes = np.unique(y)
    
    class_prior = {}
    feature_conditional = {}
    
    for cls in unique_classes:
        # Compute class prior p(y)
        mask = (y == cls)
        class_prior[cls] = np.mean(mask)
        
        # Compute feature conditionals p(x_i=1 | y)
        X_cls = X[mask]
        # For binary features, p(x_i=1|y) is just the mean of that feature in class y
        feature_conditional[cls] = np.mean(X_cls, axis=0)
    
    params = {
        'class_prior': class_prior,
        'feature_conditional': feature_conditional
    }
    
    return params


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
    class_prior = params['class_prior']
    feature_conditional = params['feature_conditional']
    
    posteriors = {}
    
    for cls in class_prior.keys():
        # Compute log posterior to avoid numerical underflow
        log_posterior = np.log(class_prior[cls])
        
        # Get p(x_i=1|y) for all features
        p_feature_given_class = feature_conditional[cls]
        
        # For each feature, compute p(x_i | y)
        for i in range(len(x)):
            if x[i] == 1:
                # p(x_i=1 | y)
                prob = p_feature_given_class[i]
            else:
                # p(x_i=0 | y) = 1 - p(x_i=1 | y)
                prob = 1 - p_feature_given_class[i]
            
            # Add to log posterior (log of product = sum of logs)
            # Handle edge case of prob = 0
            if prob > 0:
                log_posterior += np.log(prob)
            else:
                log_posterior += -np.inf
        
        posteriors[cls] = log_posterior
    
    # Find class with maximum posterior
    # In case of tie, prefer +1
    max_posterior = max(posteriors.values())
    for cls in sorted(posteriors.keys(), reverse=True):  # Check +1 first
        if posteriors[cls] == max_posterior:
            return int(cls)
    
    return 1  # Default to +1


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
    class_prior = params['class_prior']
    feature_conditional = params['feature_conditional']
    
    unnormalized_posteriors = {}
    
    for cls in class_prior.keys():
        # Compute log of unnormalized posterior
        log_posterior = np.log(class_prior[cls])
        
        p_feature_given_class = feature_conditional[cls]
        
        for i in range(len(x)):
            if x[i] == 1:
                prob = p_feature_given_class[i]
            else:
                prob = 1 - p_feature_given_class[i]
            
            if prob > 0:
                log_posterior += np.log(prob)
            else:
                log_posterior += -np.inf
        
        # Convert back from log space
        unnormalized_posteriors[cls] = np.exp(log_posterior)
    
    # Normalize
    total = sum(unnormalized_posteriors.values())
    
    if total == 0:
        return 0.5  # If both are 0, return uniform
    
    # Return p(y=+1 | x)
    return unnormalized_posteriors.get(1, 0.0) / total


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
    # Train with all features
    params_full = estimate_nb_params(X, y)
    
    # Make predictions with all features
    preds_full = np.array([predict_nb(x, params_full) for x in X])
    
    # Remove first feature (x1 is column 0)
    X_reduced = X[:, 1:]
    
    # Train with reduced features
    params_reduced = estimate_nb_params(X_reduced, y)
    
    # Make predictions with reduced features
    preds_reduced = np.array([predict_nb(x, params_reduced) for x in X_reduced])
    
    return preds_full, preds_reduced


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
    X, y = load_email_data(csv_path)
    
    preds_full, preds_reduced = drop_feature_and_retrain(X, y)
    
    # Compute accuracies
    acc_full = np.mean(preds_full == y)
    acc_reduced = np.mean(preds_reduced == y)
    
    if acc_reduced > acc_full:
        return "improves"
    elif acc_reduced < acc_full:
        return "degrades"
    else:
        return "stays the same"


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