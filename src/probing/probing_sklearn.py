import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset

from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from .abstract_probe import AbstractProbe
from ..utils import SupervisedDataset

class ProbingSklearn(AbstractProbe):
    def __init__(self,
                 probe_type: str = 'rr',
                 knn_k: int = 50,
                 device: str = 'cpu',
                 mb_size: int = 512,
                 seed: int = 42,
                 config_save_pth: str = None
                 ):
        
        self.probe_type = probe_type
        self.knn_k = knn_k
        self.device = device
        self.mb_size = mb_size
        self.seed = seed

        self.criterion = nn.CrossEntropyLoss()

        if config_save_pth is not None:
            # Save model configuration
            with open(config_save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- PROBE CONFIG ----\n')
                f.write(f'Probing type: {probe_type}\n')
                if probe_type == 'knn':
                    f.write(f'KNN k: {knn_k}\n')
                f.write(f'Eval MB size: {mb_size}\n')

    def get_name(self):
        return self.probe_type

    def probe(self,
              encoder: nn,
              tr_dataset: Dataset,
              test_dataset: Dataset,
              val_dataset: Dataset = None,
              exp_idx: int = None, # Task index on which probing is executed, if None, we are in joint or upto probing
              tr_samples_ratio: float = 1.0,
              save_file: str = None,
              ):
        
        self.encoder = encoder.to(self.device)
        self.exp_idx = exp_idx
        self.save_file = save_file
        self.tr_samples_ratio = tr_samples_ratio

        if self.save_file is not None:
            with open(self.save_file, 'a') as f:
                # Write header for probing log file
                if not os.path.exists(self.save_file) or os.path.getsize(self.save_file) == 0:
                    if self.exp_idx is not None:
                        f.write('probing_exp_idx,val_acc,test_acc\n')
                    else:
                        f.write(f'val_acc,test_acc\n')
        
        # Prepare dataloaders
        # Select only a random ratio of the train data for probing
        used_ratio_samples = int(len(tr_dataset) * self.tr_samples_ratio)
        tr_dataset, _ = random_split(tr_dataset, [used_ratio_samples, len(tr_dataset) - used_ratio_samples],
                                     generator=torch.Generator().manual_seed(self.seed)) # Generator to ensure same splits

        tr_dataset = SupervisedDataset(tr_dataset)
        train_loader = DataLoader(dataset=tr_dataset, batch_size=self.mb_size, shuffle=True, num_workers=8)
        test_dataset = SupervisedDataset(test_dataset)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.mb_size, shuffle=False, num_workers=8)
        if val_dataset is not None:
            val_dataset = SupervisedDataset(val_dataset)
            val_loader = DataLoader(dataset=val_dataset, batch_size=self.mb_size, shuffle=False, num_workers=8)

        with torch.no_grad():

            # Put encoder in eval mode, as even with no gradient it could interfere with batchnorm
            self.encoder.eval()

            # Get encoder activations for tr dataloader
            tr_activations_list = []
            tr_labels_list = []
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                activations = self.encoder(inputs)
                tr_activations_list.append(activations.detach().cpu())
                tr_labels_list.append(labels.detach().cpu())
            tr_activations = torch.cat(tr_activations_list, dim=0).numpy()
            tr_labels = torch.cat(tr_labels_list, dim=0).numpy()

            if val_dataset is not None:
                # Get encoder activations for val dataloader
                val_activations_list = []
                val_labels_list = []
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    activations = self.encoder(inputs)
                    val_activations_list.append(activations.detach().cpu())
                    val_labels_list.append(labels.detach().cpu())
                val_activations = torch.cat(val_activations_list, dim=0).numpy()
                val_labels = torch.cat(val_labels_list, dim=0).numpy()

            # Get encoder activations for test dataloader
            test_activations_list = []
            test_labels_list = []
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                activations = self.encoder(inputs)
                test_activations_list.append(activations.detach().cpu())
                test_labels_list.append(labels.detach().cpu())
            test_activations = torch.cat(test_activations_list, dim=0).numpy()
            test_labels = torch.cat(test_labels_list, dim=0).numpy()

        scaler = StandardScaler()
        
        # Use a simple classifier on activations
        if self.probe_type == 'rr':
            clf = RidgeClassifier()
        elif self.probe_type == 'knn':
            clf = KNeighborsClassifier(n_neighbors=self.knn_k)

        tr_activations = scaler.fit_transform(tr_activations)
        clf = clf.fit(tr_activations, tr_labels)
        
        if val_dataset is not None:
            # Predict validation
            val_activations = scaler.transform(val_activations)
            val_preds = clf.predict(val_activations)
            # Calculate validation accuracy and loss
            val_acc = accuracy_score(val_labels, val_preds)

        # Predict test
        test_activations = scaler.transform(test_activations)
        test_preds = clf.predict(test_activations)
        # Calculate test accuracy
        test_acc = accuracy_score(test_labels, test_preds)

        if self.save_file is not None:
            with open(self.save_file, 'a') as f:
                if val_dataset is None:
                    if self.exp_idx is not None:
                        f.write(f'{self.exp_idx},_,{test_acc:.4f}\n')
                    else:
                        f.write(f'_,{test_acc:.4f}\n')
                else:        
                    if self.exp_idx is not None:
                        f.write(f'{self.exp_idx},{val_acc:.4f},{test_acc:.4f}\n')
                    else:
                        f.write(f'{val_acc:.4f},{test_acc:.4f}\n')
