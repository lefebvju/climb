import gc
import os
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
from avalanche.benchmarks.utils import TaskAwareSupervisedClassificationDataset
from avalanche.evaluation.metrics import TaskAwareAccuracy
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

from .abstract_probe import AbstractProbe
from .metrics import MatrixMetrics
from ..utils import SupervisedDataset
from src.logger import get_writer, init_writer
writer=None
class ProbingPytorch(AbstractProbe):
    def __init__(self,                 
                 device: str = 'cpu',
                 mb_size: int = 512,
                 seed: int = 42,
                 config_save_pth: str = None,
                 dim_encoder_features: int = 512,
                 lr: float = 5e-2,
                 lr_patience: int = 5,
                 lr_factor: int = 3,
                 lr_min: float = 1e-4,
                 probing_epochs: int = 100
    ):
        
        global writer
        writer = get_writer()
        self.device = device
        self.mb_size = mb_size
        self.seed = seed
        self.probe_type = "torch"
        self.dim_encoder_features = dim_encoder_features

        self.lr = lr
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self.probing_epochs = probing_epochs

        self.criterion = nn.CrossEntropyLoss()

        if config_save_pth is not None:
            # Save model configuration
            with open(config_save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- PROBE CONFIG ----\n')
                f.write(f'Probing type: {self.probe_type}\n')
                f.write(f'Eval MB size: {mb_size}\n')
                f.write(f'Probing LR: {self.lr}\n')
                f.write(f'Probing lr patience: {self.lr_patience}\n')
                f.write(f'Probing lr factor: {self.lr_factor}\n')
                f.write(f'Probing lr min: {self.lr_min}\n')
                f.write(f'Probing epochs: {self.probing_epochs}\n')

    def get_name(self) -> str:
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

        if val_dataset is None:
            raise ValueError("Validation dataset is required for PyTorch linear probing")
        
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
        
        # Prepare datasets
        # Select only a random ratio of the train data for probing
        used_ratio_samples = int(len(tr_dataset) * self.tr_samples_ratio)
        tr_dataset, _ = random_split(tr_dataset, [used_ratio_samples, len(tr_dataset) - used_ratio_samples],
                                     generator=torch.Generator().manual_seed(self.seed)) # Generator to ensure same splits
        tr_dataset = SupervisedDataset(tr_dataset)
        test_dataset = SupervisedDataset(test_dataset)
        if val_dataset is not None:
            val_dataset = SupervisedDataset(val_dataset)

        # Put encoder in eval mode, as even with no gradient it could interfere with batchnorm
        self.encoder.eval()

        # Extract activations one loader at a time to avoid having 3×8 workers open simultaneously
        use_pin_memory = self.device != 'cpu'
        num_workers = min(os.cpu_count() or 4,12)

        def extract_activations(dataset, shuffle, split_name=''):
            loader = DataLoader(dataset=dataset, batch_size=self.mb_size, shuffle=shuffle,
                                num_workers=num_workers, pin_memory=use_pin_memory)
            acts_list, labels_list = [], []
            for inputs, labels in tqdm(loader, desc=f'Extracting {split_name}', leave=False):
                inputs = inputs.to(self.device, non_blocking=True)
                activations = self.encoder(inputs)
                acts_list.append(activations.detach().cpu())
                labels_list.append(labels)
                del inputs, activations
            del loader
            acts = nn.functional.normalize(torch.cat(acts_list, dim=0))
            labels = torch.cat(labels_list, dim=0)
            del acts_list, labels_list
            return acts, labels

        with torch.no_grad():
            tr_activations, tr_labels = extract_activations(tr_dataset, shuffle=True, split_name='train')
            val_activations, val_labels = extract_activations(val_dataset, shuffle=False, split_name='val')
            test_activations, test_labels = extract_activations(test_dataset, shuffle=False, split_name='test')

        # Labels are small (~5 MB for 1.28M samples) — always move to GPU
        tr_labels = tr_labels.to(self.device)
        val_labels = val_labels.to(self.device)
        test_labels = test_labels.to(self.device)

        torch.cuda.empty_cache()

        # Try to move activations to GPU for maximum speed; fall back to pinned CPU on OOM
        try:
            if self.device != 'cpu':
                tr_activations = tr_activations.to(self.device)
                val_activations = val_activations.to(self.device)
                test_activations = test_activations.to(self.device)
                print('Activations stored on GPU.')
        except RuntimeError:
            print('Not enough VRAM for activations, keeping on CPU (pinned).')

        torch.cuda.empty_cache()

        num_classes = len(torch.unique(tr_labels))
        if max(torch.unique(tr_labels)) > num_classes - 1:
            # If only a subset of labels, rename them in [0, num_class] range
            unique_labels = torch.unique(tr_labels)
            label_map = {k.item(): v for v, k in enumerate(unique_labels)}
            # Vectorised remap via lookup table (labels always on GPU)
            max_label = unique_labels.max().item() + 1
            lut = torch.zeros(max_label, dtype=torch.long, device=self.device)
            for orig, new in label_map.items():
                lut[orig] = new
            tr_labels = lut[tr_labels]
            val_labels = lut[val_labels]
            test_labels = lut[test_labels]


        # Set up Linear Probe
        linear_probe_clf = SSLEvaluator(self.dim_encoder_features, num_classes, 0, 0.0)
        linear_probe_clf.to(self.device)
        _lr = self.lr 
        linear_probe_clf_optimizer = torch.optim.Adam(linear_probe_clf.parameters(), lr=_lr)

        classifier_train_step = 0
        val_step = 0
        best_val_loss = 1e10
        best_val_acc = 0.0
        patience = self.lr_patience
        linear_probe_clf.train()
        best_model = None
        
        # Training loop of the probe
        for e in range(self.probing_epochs):
            # Shuffle consistently: perm on GPU (labels), index into activations (CPU or GPU)
            perm = torch.randperm(len(tr_labels), device=self.device)
            tr_activations = tr_activations[perm.to(tr_activations.device)]
            tr_labels = tr_labels[perm]

            train_loss = 0.0
            train_samples = 0.0
            index = 0
            while index + self.mb_size <= len(tr_labels):
                idx_end = min(index + self.mb_size, len(tr_labels))
                _x = tr_activations[index:idx_end, :].to(self.device, non_blocking=True)
                y = tr_labels[index:idx_end]

                # forward pass
                mlp_preds = linear_probe_clf(_x)
                mlp_loss = self.criterion(mlp_preds, y)
                # update finetune weights
                mlp_loss.backward()
                linear_probe_clf_optimizer.step()
                linear_probe_clf_optimizer.zero_grad()
                train_loss += mlp_loss.item()
                train_samples += len(y)


                classifier_train_step += 1
                index = idx_end

            # Eval on validation sets
            linear_probe_clf.eval()
            val_loss = 0.0
            acc_correct = 0
            acc_all = 0
            with torch.no_grad():
                singelite = False if len(val_activations) > self.mb_size else True
                index = 0
                while index < len(val_activations) or singelite:
                    idx_end = min(index + self.mb_size, len(val_activations))
                    _x = val_activations[index:idx_end, :].to(self.device, non_blocking=True)
                    y = val_labels[index:idx_end]
                    # forward pass
                    mlp_preds = linear_probe_clf(_x)
                    mlp_loss = F.cross_entropy(mlp_preds, y)
                    val_loss += mlp_loss.item()
                    n_corr = (mlp_preds.argmax(1) == y).sum().cpu().item()
                    n_all = y.size()[0]
                    _val_acc = n_corr / n_all
                    acc_correct += n_corr
                    acc_all += n_all
                    val_step += 1
                    index = idx_end
                    singelite = False
            
             # mean validation loss
            val_loss = val_loss / acc_all
            val_acc = acc_correct / acc_all

            print(
            f'| Epoch {e} | Train loss: {train_loss:.6f} | Valid loss: {val_loss:.6f} acc: {100 * val_acc:.2f} |'
            )
            
            # Adapt lr
            if val_acc > best_val_acc or best_model is None:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_model = copy.deepcopy(linear_probe_clf.model.state_dict())
                patience = self.lr_patience
                print('*', end='', flush=True)
            else:
                patience -= 1
                if patience <= 0:
                    _lr /= self.lr_factor
                    print(' lr={:.1e}'.format(_lr),)
                    if _lr < self.lr_min:
                        print(' NO MORE PATIENCE')
                        break
                    patience = self.lr_patience
                    linear_probe_clf_optimizer.param_groups[0]['lr'] = _lr
                    linear_probe_clf.model.load_state_dict(best_model)

        linear_probe_clf.model.load_state_dict(best_model)
        linear_probe_clf.eval()

        if isinstance(test_dataset,TaskAwareSupervisedClassificationDataset) or isinstance(test_dataset.data,TaskAwareSupervisedClassificationDataset) :
            joint=False
        else:
            joint=True
            try:
                labels_by_task=[[label_map[label] for label in list(test_dataset.data.datasets[i].targets.count.keys())] for i in range(len(test_dataset.data.datasets))]
            except NameError:
                labels_by_task=[list(test_dataset.data.datasets[i].targets.count.keys()) for i in range(len(test_dataset.data.datasets))]
                
            matrix_metrics = MatrixMetrics(labels_by_task,save_file.split("probing_ratio")[0]+"matrix.csv",exp_idx=int(save_file.split("_")[-1].split(".csv")[0]))

        # Eval on test set
        with torch.no_grad():
            test_loss = 0.0
            acc_correct = 0
            acc_all = 0
            singelite = False if len(test_activations) > self.mb_size else True
            index = 0
            while index < len(test_activations) or singelite:
                idx_end = min(index + self.mb_size, len(test_activations))
                _x = test_activations[index:idx_end, :].to(self.device, non_blocking=True)
                y = test_labels[index:idx_end]
                # forward pass
                mlp_preds = linear_probe_clf(_x)
                mlp_loss = F.cross_entropy(mlp_preds, y)
                test_loss += mlp_loss.item()
                if joint:
                    matrix_metrics.update(mlp_preds,y,exp_idx)
                n_corr = (mlp_preds.argmax(1) == y).sum().cpu().item()
                n_all = y.size()[0]
                _test_acc = n_corr / n_all
                acc_correct += n_corr
                acc_all += n_all
                index = idx_end
                singelite = False
        
            # mean test loss
            test_loss = val_loss / acc_all
            test_acc = acc_correct / acc_all
            if joint:
                res= matrix_metrics.result()
                

            print(f'Test loss: {test_loss}, test acc: {test_acc}')

        if self.save_file is not None:
            with open(self.save_file, 'a') as f:
                if val_dataset is None:
                    if self.exp_idx is not None:
                        f.write(f'{self.exp_idx},_,{test_acc:.4f}\n')
                    else:
                        f.write(f'_,{test_acc:.4f}\n')
                else:
                    if self.exp_idx is not None:
                        f.write(f'{self.exp_idx},{best_val_acc:.4f},{test_acc:.4f}\n')
                    else:
                        f.write(f'{best_val_acc:.4f},{test_acc:.4f}\n')

        del tr_activations, val_activations, test_activations
        del tr_labels, val_labels, test_labels
        del linear_probe_clf, linear_probe_clf_optimizer, best_model
        self.encoder = None
        gc.collect()
        torch.cuda.empty_cache()



class SSLEvaluator(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.out_features = n_classes  # for *head* compability
        if n_hidden is None or n_hidden == 0:
            # use linear classifier
            self.model = nn.Sequential(nn.Flatten(), nn.Dropout(p=p), nn.Linear(n_input, n_classes, bias=True))
        else:
            # use simple MLP classifier
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True),
            )

    def forward(self, x):
        logits = self.model(x)
        return logits