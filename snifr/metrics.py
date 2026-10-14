"""Evaluation Metrics for Child Harmful Content Detection"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def compute_detection_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Compute detection metrics.

    Args:
        predictions: Predicted category IDs
        targets: Ground truth category IDs

    Returns:
        Dictionary with accuracy, precision, recall, F1
    """
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted', zero_division=0)
    recall = recall_score(targets, predictions, average='weighted', zero_division=0)
    f1 = f1_score(targets, predictions, average='weighted', zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def compute_confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute confusion matrix"""
    return confusion_matrix(targets, predictions)
