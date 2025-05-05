from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import List, Dict, Any, Union

import numpy as np

### We stole these metrics from our IML project, still needs to be updated according to our needs but here is a template :))

class ModelMetrics:
    """
    Wrapper class for model evaluation metrics.
    This class provides methods for calculating accuracy
    F1-score, and confusion matrix.
    """

    def __init__(self, y_true: Union[
            List[int], np.ndarray], y_pred: Union[List[int], np.ndarray]
            ) -> None:
        """
        Initializes the ModelMetrics object with true and predicted labels.

        Args:
        - y_true: The true labels
        - y_pred: The predicted labels
        """
        self.y_true = y_true
        self.y_pred = y_pred

    def accuracy(self) -> float:
        """
        Computes the accuracy of the model.

        Returns:
        - accuracy: The accuracy score
        """
        return accuracy_score(self.y_true, self.y_pred)

    def f1_score(self, average: str = 'weighted') -> float:
        """
        Computes the F1-score of the model.

        Args:
        - average: The type of averaging for multi-class
          classification (default is 'weighted')

        Returns:
        - f1: The F1 score
        """
        return f1_score(self.y_true, self.y_pred, average=average)

    def confusion_matrix(self) -> np.ndarray:
        """
        Computes the confusion matrix for the model.

        Returns:
        - cm: The confusion matrix
        """
        return confusion_matrix(self.y_true, self.y_pred)

    def print_metrics(self) -> None:
        """
        Prints out accuracy, F1-score, and confusion matrix.

        Returns:
        - None
        """
        print("Accuracy: {:.4f}".format(self.accuracy()))
        print("F1-Score: {:.4f}".format(self.f1_score()))
        print("Confusion Matrix:")
        print(self.confusion_matrix())

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing all metrics.

        Returns:
        - metrics: A dictionary containing accuracy, F1-score,
          and confusion matrix
        """
        return {
            'accuracy': self.accuracy(),
            'f1_score': self.f1_score(),
            'confusion_matrix': self.confusion_matrix()
        }