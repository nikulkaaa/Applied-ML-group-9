from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import List, Dict, Any, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_auc_score, roc_curve
)
from typing import List, Dict, Any, Union, Optional
import numpy as np


class ModelMetrics:
    """
    Wrapper class for model evaluation metrics.
    This class provides methods for calculating accuracy,
    F1-score, confusion matrix, ROC-AUC, and EER.
    """

    def __init__(
        self,
        y_true: Union[List[int], np.ndarray],
        y_pred: Union[List[int], np.ndarray],
        y_scores: Optional[Union[List[float], np.ndarray]] = None
    ) -> None:
        """
        Initializes the ModelMetrics object with true and predicted labels.

        Args:
        - y_true: The true labels
        - y_pred: The predicted labels
        - y_scores: The predicted scores/probabilities (required for ROC-AUC and EER)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_scores = np.array(y_scores) if y_scores is not None else None

    def accuracy(self) -> float:
        return accuracy_score(self.y_true, self.y_pred)

    def f1_score(self, average: str = 'weighted') -> float:
        return f1_score(self.y_true, self.y_pred, average=average)

    def confusion_matrix(self) -> np.ndarray:
        return confusion_matrix(self.y_true, self.y_pred)

    def roc_auc(self) -> Optional[float]:
        if self.y_scores is None:
            return None
        try:
            return roc_auc_score(self.y_true, self.y_scores)
        except ValueError:
            return None  # In case of multiclass without proper scoring

    def eer(self) -> Optional[float]:
        """
        Computes the Equal Error Rate (EER) for binary classification.
        Returns None if y_scores are not provided or if y_true is not binary.
        """
        if self.y_scores is None or len(np.unique(self.y_true)) != 2:
            return None

        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_scores)
        fnr = 1 - tpr
        eer_threshold_idx = np.nanargmin(np.absolute((fnr - fpr)))
        eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2
        return eer

    def print_metrics(self) -> None:
        print("Accuracy: {:.4f}".format(self.accuracy()))
        print("F1-Score: {:.4f}".format(self.f1_score()))
        print("Confusion Matrix:")
        print(self.confusion_matrix())
        if self.roc_auc() is not None:
            print("ROC-AUC: {:.4f}".format(self.roc_auc()))
        else:
            print("ROC-AUC: Not Available (requires scores)")
        if self.eer() is not None:
            print("EER: {:.4f}".format(self.eer()))
        else:
            print("EER: Not Available (binary classification with scores required)")

    def get_all_metrics(self) -> Dict[str, Any]:
        return {
            'accuracy': self.accuracy(),
            'f1_score': self.f1_score(),
            'confusion_matrix': self.confusion_matrix(),
            'roc_auc': self.roc_auc(),
            'eer': self.eer()
        }
