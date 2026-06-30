import torch
from torch import Tensor
from avalanche.evaluation import Metric
from avalanche.evaluation.metrics import Accuracy
import os
import pandas as pd
import numpy as np
from src.logger import get_writer

writer=None

class MatrixMetrics(Metric[float]):
    """
    This metric calculates the accuracy for each task at each `exp_id`.
    """

    def __init__(self, labels_by_task, file_path,exp_idx=None):
        """
        Initialize the matrix of metrics.

        :param labels_by_task: list of lists of labels per task
        :param file_path: path to a file, used to save results (optional)
        """
        super().__init__()
        
        global writer
        writer = get_writer()
        self.file_path = file_path
        self.labels_by_task = labels_by_task  # List of tasks with their labels
        self._accuracy_metric = [Accuracy() for _ in range(len(labels_by_task))]
        self.label_to_task = {}
        self.exp_idx = exp_idx

        # Create the label -> task mapping
        for task_id, labels in enumerate(labels_by_task):
            for label in labels:
                self.label_to_task[label] = task_id


        # Initialize a matrix to store results (exp_id x task_id)
        self.accuracy_matrix = []

    def update(self, predicted_y: Tensor, true_y: Tensor, exp_id: int):
        """
        Update accuracy for each task and exp_id.

        :param predicted_y: Model predictions
        :param true_y: Ground truth labels
        :param exp_id: Experiment ID (or batch ID)
        """
        # Ensure inputs are tensors
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch between true_y and predicted_y tensors.")

        # Associate each label with its task
        for true, pred in zip(true_y, predicted_y):
            task_id = self.label_to_task[int(true)]  # Find the corresponding task
            # Update accuracy for this task
            self._accuracy_metric[task_id].update(pred.unsqueeze(dim=0), true.unsqueeze(dim=0))

        # Calculate accuracy for each task and store it in the matrix for this exp_id
        task_accuracies = [metric.result() for metric in self._accuracy_metric]

        # Store the results for this exp_id
        self.accuracy_matrix = task_accuracies

    def append_column(self, csv_path, values, col_name="new_col"):
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if len(df) != len(values):
                raise ValueError("Longueur différente entre CSV et liste.")
            df[col_name] = values
        else:
            df = pd.DataFrame({col_name: values})

        
        df.to_csv(csv_path, index=False)
        if self.exp_idx >0:
            self.result_df = self.calculate_learning_metrics(df)
        else:
            self.result_df = {
                'CA': 0,
                'CA_all': 0,
                'FA': 0,
                'F_max': 0,
                'F_first': 0,
                'F_max_relat': 0,
                'F_first_relat': 0,
                'FT': 0
            }
        metrics_df = pd.DataFrame([self.result_df])
        metrics_df.to_csv(csv_path.split("matrix.cs")[0]+'result.csv', index=False)
            
    def result(self):
        """
        Returns the complete accuracy matrix, where each row is a task and each
        column is an `exp_id` (experiment ID).

        :return: Dictionary with exp_id as the key and the list of accuracies per task as value
        """
        if self.exp_idx >=0:
            self.append_column(self.file_path, self.accuracy_matrix, self.exp_idx)
        return self.accuracy_matrix

    def reset(self, **kwargs):
        """
        Reset your metric here
        """
        for t in range(len(self.labels_by_task)):
            self._accuracy_metric[t].reset()
        self.accuracy_matrix = {}

    def calculate_learning_metrics(self,df):
        """
        Calculate comprehensive learning and forgetting metrics from a DataFrame.
    
        Parameters:
        -----------
        df : pandas.DataFrame
            Matrix where rows represent training steps and columns represent tasks.
            Can be rectangular (more rows than columns). Values represent performance metrics.
    
        Returns:
        --------
        dict
            Dictionary containing the following metrics:
            - CA: Average performance on upper triangular (continual accuracy)
            - CA_all: Average performance on all entries
            - FA: Final average performance (last column of upper triangle)
            - F_max: Absolute forgetting from maximum performance
            - F_first: Absolute forgetting from diagonal (first performance)
            - F_max_relat: Relative forgetting from maximum (percentage)
            - F_first_relat: Relative forgetting from diagonal (percentage)
            - FT: Average performance on lower triangular (backward transfer)
        """
        # Create masks once - handle rectangular DataFrames
        n_rows, n_cols = df.shape
        upper_mask = np.triu(np.ones((n_rows, n_cols), dtype=bool))
        lower_strict_mask = np.tril(np.ones((n_rows, n_cols), dtype=bool), k=-1)
    
        # Get values array once for reuse
        df_values = df.values
    
        # Apply masks
        upper_df = df.where(upper_mask)
        lower_strict_df = df.where(lower_strict_mask)
    
        # Extract diagonal (limited by minimum dimension)
        diag = np.diag(df_values[:min(n_rows, n_cols), :min(n_rows, n_cols)])
    
        # Compute statistics efficiently
        avg_eval_upper = upper_df.mean(axis=0)
        avg_eval_lower = lower_strict_df.mean(axis=0)
        avg_eval = df.mean(axis=0)
        max_training = upper_df.max(axis=1)
    
        # Extract last column once
        last_col = upper_df.iloc[:, -1]
    
        # Calculate metrics
        CA = avg_eval_upper.mean()
        CA_all = avg_eval.mean()
        FA = avg_eval_upper.iloc[-1]
    
        # Forgetting metrics - computed together to avoid redundant operations
        # Only compute for rows where diagonal exists (first n_cols rows)
        n_diag = len(diag)
        max_diff = max_training[:n_diag] - last_col[:n_diag]
        diag_diff = diag - last_col[:n_diag]
    
        F_max = max_diff.mean()
        F_first = diag_diff.mean()
        F_max_relat = (max_diff / max_training[:n_diag]).mean() * 100
        F_first_relat = (diag_diff / diag).mean() * 100
    
        FT = avg_eval_lower.mean()

        writer.add_scalar('metrics/CA', CA, self.exp_idx)
        writer.add_scalar('metrics/CA_all', CA_all, self.exp_idx)
        writer.add_scalar('metrics/FA', FA, self.exp_idx)
        writer.add_scalar('metrics/F_max', F_max, self.exp_idx)
        writer.add_scalar('metrics/F_first', F_first, self.exp_idx)
        writer.add_scalar('metrics/F_max_relat', F_max_relat, self.exp_idx)
        writer.add_scalar('metrics/F_first_relat', F_first_relat, self.exp_idx)
        writer.add_scalar('metrics/FT', FT, self.exp_idx)
        writer.add_scalar('metrics/FA_all', avg_eval.iloc[-1], self.exp_idx)
        
    
        return {
            'CA': CA,
            'CA_all': CA_all,
            'FA': FA,
            'F_max': F_max,
            'F_first': F_first,
            'F_max_relat': F_max_relat,
            'F_first_relat': F_first_relat,
            'FT': FT
        }