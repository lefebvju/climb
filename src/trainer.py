import os
import random
from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import DataLoader

from .utils import UnsupervisedDataset
from .transforms import get_transforms
from .ssl_models import AbstractSSLModel
from .strategies import AbstractStrategy
from .optims import init_optim
from .probing import exec_probing
from src.logger import get_writer, init_writer
from src.logger import logger
writer=None

class Trainer():

    def __init__(self,
                 ssl_model: AbstractSSLModel = None,
                 strategy: AbstractStrategy = None,
                 optim: str = 'SGD',
                 lr: float = 0.01,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4,
                 lars_eta: float = 0.005,
                 train_mb_size: int = 32,
                 train_epochs: int = 1,
                 mb_passes: int = 3,
                 device = 'cpu',
                 dataset_name: str = 'cifar100',
                 save_pth: str  = None,
                 save_model: bool = False,
                 common_transforms: bool = True,
                 num_views: int = 2,
                 latent_log_every: int = 50,
               ):
        global writer
        writer = get_writer()
        
        if ssl_model is None:
            raise Exception(f'A SSL model is requred')            

        self.ssl_model = ssl_model
        self.strategy = strategy
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lars_eta = lars_eta
        self.train_mb_size = train_mb_size
        self.train_epochs = train_epochs
        self.mb_passes = mb_passes
        self.device = device
        self.dataset_name = dataset_name
        self.save_pth = save_pth
        self.save_model = save_model
        self.common_transforms = common_transforms
        self.num_views = num_views # == 2 for most Instance Discrimination methods, but can vary e.g. EMP
        self.latent_log_every = latent_log_every

        self.model_and_strategy_name = self.strategy.get_name() + '_' + self.ssl_model.get_name()

        # Set up transforms
        if self.common_transforms:
            self.transforms = get_transforms(dataset=self.dataset_name, model='common', n_crops=num_views)
        else:
            self.transforms = get_transforms(dataset=self.dataset_name, model=self.ssl_model.get_name(), n_crops=num_views)

        # List of params to optimize
        params_to_optimize = self.ssl_model.get_params() + self.strategy.get_params()

        # Set up optimizer
        self.optimizer = init_optim(optim, params_to_optimize, lr=self.lr, momentum=self.momentum,
                                    weight_decay=self.weight_decay, lars_eta=self.lars_eta)
        self.cbp=0
        self.step=0
        self.step_batch=0

        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- TRAINER CONFIG ----\n')
                f.write(f'optim: {optim}\n') 
                f.write(f'Learning Rate: {self.lr}\n')
                f.write(f'optim-momentum: {self.momentum}\n')
                f.write(f'weight_decay: {self.weight_decay}\n')
                if optim == 'lars':
                    f.write(f'lars_eta: {self.lars_eta}\n')
                f.write(f'num_views: {self.num_views}\n')
                f.write(f'train_mb_size: {self.train_mb_size}\n')
                f.write(f'train_epochs: {self.train_epochs}\n')
                f.write(f'mb_passes: {self.mb_passes}\n')


                # Write loss file column names
                with open(os.path.join(self.save_pth, 'pretr_loss.csv'), 'a') as f:
                    f.write('loss,exp_idx,epoch,mb_idx,mb_pass\n')


    def train_experience(self, 
                         dataset,
                         exp_idx: int,
                         iid_intermediate_eval_dict: dict = {"status": False}, # Set to True to evaluate model at intermediate steps, contains vars for intermediate eval
                         ):
        # Prepare data

        exp_data = UnsupervisedDataset(dataset)
        data_loader = DataLoader(exp_data, batch_size=self.train_mb_size, shuffle=True, drop_last=True, num_workers=12)


        if iid_intermediate_eval_dict["status"]:
            # Calculate number of total training steps
            tot_tr_steps = self.train_epochs * len(data_loader) * self.mb_passes
            tr_step_idx = 0
            eval_every_steps = int(tot_tr_steps / iid_intermediate_eval_dict["num_exps"])
            eval_idx = 0

        self.ssl_model.train()
        self.strategy.train()

        self.strategy.before_experience()

        for epoch in range(self.train_epochs):
            for mb_idx, stream_mbatch in enumerate(tqdm(data_loader)):
                stream_mbatch = stream_mbatch.to(self.device)

                stream_mbatch = self.strategy.before_mb_passes(stream_mbatch)

                self.step_batch+=1
                self.step+=stream_mbatch.shape[0]

                for k in range(self.mb_passes):
                    # Apply strategy modifications before forward pass (e.g. concat replay samples from buffer)
                    mbatch = self.strategy.before_forward(stream_mbatch)
                    # Apply transforms, obtains a list of tensors, each containing 1 view for every sample in the mbatch
                    x_views_list = self.transforms(mbatch)

                    x_views_list = self.strategy.after_transforms(x_views_list)
                    self.cbp+=self.num_views*x_views_list[0].shape[0]
                    writer.add_scalar('Gradient/CBP',self.cbp,self.step)
                    # Forward pass of SSL model (z: projector features, e: encoder features)
                    loss, z_list, e_list = self.ssl_model(x_views_list)

                    # Strategy after forward pass

                    loss_strategy = self.strategy.after_forward(x_views_list, loss, z_list, e_list)

                    if loss_strategy is not None:
                        # Backward pass
                        writer.add_scalar('Loss/total_loss', loss_strategy.item(), self.step)
                        self.optimizer.zero_grad()
                        loss_strategy.backward()
                        self.optimizer.step()

                    self.ssl_model.after_backward()
                    self.strategy.after_backward()

                    del x_views_list, z_list, e_list, loss, loss_strategy, mbatch

                    # Check if have to evaluate IID model
                    if iid_intermediate_eval_dict["status"]:
                        tr_step_idx += 1
                        if tr_step_idx % eval_every_steps == 0:
                            exec_probing(kwargs=iid_intermediate_eval_dict["kwargs"], probes=iid_intermediate_eval_dict["probes"],
                                         probing_benchmark=iid_intermediate_eval_dict["benchmark"], encoder=self.ssl_model.get_encoder_for_eval(),
                                         pretr_exp_idx=eval_idx, probing_tr_ratio_arr=iid_intermediate_eval_dict["probing_tr_ratio_arr"],
                                         save_pth=self.save_pth)
                            eval_idx += 1
                    self.ssl_model.train()

                self.strategy.after_mb_passes()

        # Save model and optimizer state
        if self.save_model and self.save_pth is not None:
            chkpt_pth = os.path.join(self.save_pth, 'checkpoints')
            if not os.path.exists(chkpt_pth):
                os.makedirs(chkpt_pth)
            torch.save(self.ssl_model.state_dict()
            , os.path.join(chkpt_pth, f'model_exp{exp_idx}.pth'))

        return self.ssl_model

    def get_resume_state(self) -> dict:
        """Collect all trainer state needed to resume training."""
        state = {
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'step_batch': self.step_batch,
            'cbp': self.cbp,
            'torch_rng': torch.get_rng_state(),
            'numpy_rng': np.random.get_state(),
            'python_rng': random.getstate(),
        }
        if torch.cuda.is_available():
            state['cuda_rng'] = torch.cuda.get_rng_state()
        return state

    def load_resume_state(self, state: dict):
        """Restore trainer state from a checkpoint dict."""
        self.optimizer.load_state_dict(state['optimizer'])
        self.step = state['step']
        self.step_batch = state['step_batch']
        self.cbp = state['cbp']
        torch.set_rng_state(state['torch_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        if 'cuda_rng' in state and torch.cuda.is_available():
            torch.cuda.set_rng_state(state['cuda_rng'])
