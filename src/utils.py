import os
import numpy as np
import pandas as pd
import argparse

import torch
from torch.utils.data import Dataset
from torchvision import models

from avalanche.evaluation.metrics import Forgetting

# Convert Avalanche dataset with labels and task labels to Pytorch dataset with only input tensors
class UnsupervisedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor, _, _ = self.data[idx]
        return input_tensor
    
class SupervisedDataset(Dataset):
    def __init__(self, data, label=None):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            if len(self.data[idx]) >= 3:
                input_tensor, label, _ = self.data[idx]
            elif len(self.data[idx]) == 2:
                input_tensor, label = self.data[idx]
                
            return input_tensor, label


@torch.no_grad() 
def update_ema_params(model_params, ema_model_params, momentum):
    for po, pm in zip(model_params, ema_model_params):
            pm.data.mul_(momentum).add_(po.data, alpha=(1. - momentum))


def calculate_forgetting(save_pth, num_exps, probing_tr_ratio_arr=[1]):
     # Init forgetting
    forgetting_val = Forgetting()
    forgetting_test = Forgetting()
    for probe_tr_ratio in probing_tr_ratio_arr:
        separate_pth = os.path.join(save_pth, f'probing_separate/probing_ratio{probe_tr_ratio}')
        forgetting_folder = os.path.join(save_pth, f'forgetting/probing_ratio{probe_tr_ratio}')
        if not os.path.exists(forgetting_folder):
            os.makedirs(forgetting_folder)
        with open(os.path.join(forgetting_folder, 'forgetting.csv'), 'a') as f:
            f.write('exp_idx,val_forgetting,test_forgetting\n')
        final_df = pd.read_csv(os.path.join(separate_pth, f'probe_exp_{num_exps-1}.csv'))
        for exp_idx in range(num_exps):
            initial_df = pd.read_csv(os.path.join(separate_pth, f'probe_exp_{exp_idx}.csv'))
            # Take the row where probing_exp_idx = exp_idx  
            initial_score_val = initial_df[initial_df['probing_exp_idx'] == exp_idx]['val_acc'].values[0]
            initial_score_test = initial_df[initial_df['probing_exp_idx'] == exp_idx]['test_acc'].values[0]
            forgetting_val.update_initial(k=exp_idx, v=initial_score_val)
            forgetting_test.update_initial(k=exp_idx, v=initial_score_test)
            final_score_val = final_df[final_df['probing_exp_idx'] == exp_idx]['val_acc'].values[0]
            final_score_test = final_df[final_df['probing_exp_idx'] == exp_idx]['test_acc'].values[0]
            forgetting_val.update_last(k=exp_idx, v=final_score_val)
            forgetting_test.update_last(k=exp_idx, v=final_score_test)

            with open(os.path.join(forgetting_folder, 'forgetting.csv'), 'a') as f:
                f.write(f'{exp_idx},{forgetting_val.result()[exp_idx]},{forgetting_test.result()[exp_idx]}\n')

        with open(os.path.join(forgetting_folder, 'avg_forgetting.csv'), 'a') as f:
            f.write('val_avg_forgetting,test_avg_forgetting\n')
            avg_val = sum(forgetting_val.result().values()) / len(forgetting_val.result().values())
            avg_test = sum(forgetting_test.result().values()) / len(forgetting_test.result().values())
            f.write(f'{avg_val},{avg_test}\n')

def save_avg_stream_acc(probe, save_pth):
    """
        Calculate and save avg joint accuracy across the stream.
    """
    probing_folder = os.path.join(save_pth, f'probe_{probe}/probing_joint/probing_ratio1')

    val_acc_list, test_acc_list = [], []
    for file in os.listdir(probing_folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(probing_folder, file))
            val_acc_list.append(df['val_acc'].values[0])
            test_acc_list.append(df['test_acc'].values[0])

    avg_val_acc = np.mean(val_acc_list)
    avg_test_acc = np.mean(test_acc_list)

    output_file = os.path.join(save_pth, f"avg_stream_acc.csv")
    with open(output_file, "a") as output_f:
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            output_f.write("probe_type,avg_val_acc,avg_test_acc\n")
        output_f.write(f"{probe},{avg_val_acc:.4f},{avg_test_acc:.4f}\n")
    


def write_final_scores(probe, folder_input_path, output_file):
    """
    Report final aggregated scores of the probing

    """
    # output_file = os.path.join(folder_path, "final_scores.csv")
    with open(output_file, "a") as output_f:
        # Write header
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            output_f.write("probe_type,probe_ratio,avg_val_acc,avg_test_acc\n")

        # Get all subfolder paths starting with "probing_ratio"
        probing_ratios_subfolders = [os.path.join(folder_input_path, f) for f in os.listdir(folder_input_path) 
                                    if os.path.isdir(os.path.join(folder_input_path, f)) and f.startswith("probing_ratio")]

        # For each probing tr ratio
        for subfolder in probing_ratios_subfolders:
            probing_tr_ratio = subfolder.split("probing_ratio")[1]
            probe_exp_df_list = [] # List of tuples (Dataframe, exp_index)

            # Read all csv, one for each experience on which probing has been executed
            for file in os.listdir(subfolder):
                if file.endswith('.csv'):
                    probe_exp = int(file.split('.csv')[0].split('probe_exp_')[-1]) # Finds exp_idx from filename
                    df = pd.read_csv(os.path.join(subfolder, file))
                    probe_exp_df_list.append((df, probe_exp))

            # Find df with highest exp_index in probe_exp_df_list
            final_df = max(probe_exp_df_list, key=lambda x: x[1])[0]
            # Get final test and validation accuracies
            final_avg_test_acc =  final_df['test_acc'].mean()
            final_avg_val_acc = final_df['val_acc'].mean()


            output_f.write(f"{probe},{probing_tr_ratio},{final_avg_val_acc:.4f},{final_avg_test_acc:.4f}\n")


def read_command_line_args():
    """
    Parses command line arguments
    """
    def str_to_bool(s):
        if s.lower() in ('true', 't', 'yes', 'y', '1'):
            return True
        elif s.lower() in ('false', 'f', 'no', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-idx', type=int, default=0)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset-seed', type=int, default=None)
    parser.add_argument('--strategy', type=str, default='climb')
    parser.add_argument('--model', type=str, default='simclr')
    parser.add_argument('--encoder', type=str, default='resnet18')
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--optim-momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lars-eta', type=float, default=0.005)
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--dataset-root', type=str, default='./data')
    parser.add_argument('--num-exps', type=int, default=20)
    parser.add_argument('--save-folder', type=str, default='./logs')
    parser.add_argument('--config-path', type=str, default='configs/config_climb.json')
    parser.add_argument('--dim-proj', type=int, default=2048)
    parser.add_argument('--dim-pred', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=1)
    
    parser.add_argument('--mb-passes', type=int, default=3)
    parser.add_argument('--tr-mb-size', type=int, default=10)
    parser.add_argument('--common-transforms', type=str_to_bool, default=False)#True à l'origine
    parser.add_argument('--iid', type=str_to_bool, default=False)
    parser.add_argument('--no-train', type=str_to_bool, default=False)
    parser.add_argument('--save-model-final', type=str_to_bool, default=True)
    parser.add_argument('--save-model-every-exp', type=str_to_bool, default=False)
    parser.add_argument('--blurred', type=str_to_bool, default=False)

    # Downstream task params
    parser.add_argument('--downstream', type=str_to_bool, default=False)
    parser.add_argument('--downstream-dataset', type=str, default='svhn')
    parser.add_argument('--downstream-dataset-root', type=str, default='./data')

    # Multi-processing params
    parser.add_argument('--max-process', type=int, default=1)

    # Pretrain initialization
    parser.add_argument('--pretrain-init-type', type=str, default='none')
    parser.add_argument('--pretrain-init-source', type=str, default='imagenet_1k')
    parser.add_argument('--pretrain-init-pth', type=str, default='' )

    # Datasets
    parser.add_argument('--evaluation-protocol-clear', type=str, default='iid') # iid (curr exp eval) or streaming (next exp eval)

    # Probing params
    parser.add_argument('--probing-all-exp', type=str_to_bool, default=True)
    parser.add_argument('--eval-mb-size', type=int, default=256)
    parser.add_argument('--probing-rr', type=str_to_bool, default=False)
    parser.add_argument('--probing-knn', type=str_to_bool, default=False)
    parser.add_argument('--probing-torch', type=str_to_bool, default=True)
    parser.add_argument('--probing-separate', type=str_to_bool, default=False)
    parser.add_argument('--probing-upto', type=str_to_bool, default=False)
    parser.add_argument('--probing-joint', type=str_to_bool, default=True)
    parser.add_argument('--probing-val-ratio', type=float, default=0.1)
    parser.add_argument('--use-probing-tr-ratios', type=str_to_bool, default=False)
    parser.add_argument('--knn-k', type=int, default=50)
    parser.add_argument('--probe-lr', type=float, default=5e-2)
    parser.add_argument('--probe-lr-patience', type=int, default=5)
    parser.add_argument('--probe-lr-factor', type=float, default=3.0)
    parser.add_argument('--probe-lr-min', type=float, default=1e-4)
    parser.add_argument('--probe-epochs', type=int, default=100)
    

    # Replay params
    parser.add_argument('--buffer-type', type=str, default='default')
    parser.add_argument('--mem-size', type=int, default=2500)
    parser.add_argument('--repl-mb-size', type=int, default=128)

    # CLA params
    # --omega is λ in the paper (Eq. 2): L = L_SSL + λ * L_align
    # λ=1.0 for ImageNet-100, λ=2.0 for CIFAR-100 (Section 4.4)
    parser.add_argument('--omega', type=float, default=1.0)
    parser.add_argument('--align-criterion', type=str, default='cosine')
    parser.add_argument('--use-aligner', type=str_to_bool, default=True)
    parser.add_argument('--align-after-proj', type=str_to_bool, default=True)
    parser.add_argument('--aligner-dim', type=int, default=512) # If set <= 0 it uses pred_dim instead


    # SSL models specific params
    parser.add_argument('--num-views', type=int, default=2) # Most Instance Discrimination SSL methods use 2, but can vary (e.g EMP)
    parser.add_argument('--lambd', type=float, default=5e-3) # For Barlow 
    parser.add_argument('--barlow-loss-scaling', type=float, default=0.1)
    parser.add_argument('--byol-momentum', type=float, default=0.99)
    parser.add_argument('--return-momentum-encoder', type=str_to_bool, default=True)
    parser.add_argument('--simclr-temp', type=float, default=0.5)
    
    # MAE params
    parser.add_argument('--mae-patch-size', type=int, default=2)                
    parser.add_argument('--mae-emb-dim', type=int, default=192)
    parser.add_argument('--mae-decoder-layer', type=int, default=4)                
    parser.add_argument('--mae-decoder-head', type=int, default=3)
    parser.add_argument('--mae-mask-ratio', type=float, default=0.75)

    # ViT params
    parser.add_argument('--vit-encoder-layer', type=int, default=12)
    parser.add_argument('--vit-encoder-head', type=int, default=3)
    parser.add_argument('--vit-avg-pooling', type=str_to_bool, default=False)

    # SCALE params
    parser.add_argument('--scale-dim-features', type=int, default=128)
    parser.add_argument('--scale-distill-power', type=float, default=0.15)
    parser.add_argument('--use-scale-buffer', type=str_to_bool, default=True)

    # CLIMB params
    parser.add_argument('--distill-only-old', type=str_to_bool, default=False)
    parser.add_argument('--distillation', type=str_to_bool, default=False)
    parser.add_argument('--ema', type=str_to_bool, default=False)
    parser.add_argument('--tau-ema', type=float, default=0.999)
    parser.add_argument('--ltm-merge-strategy', type=str, default="random")
    parser.add_argument('--weight-mode', type=str, default="none")
    parser.add_argument('--update-ltm-bool', type=str_to_bool, default=False)
    parser.add_argument('--eta-min', type=float, default=0.0001)
    parser.add_argument('--ratio-ltm', type=float, default=0.5)
    parser.add_argument('--pruning', type=str, default="no_reset")
    parser.add_argument('--stm-size', type=int, default=100)
    parser.add_argument('--max-examples-per-centroid', type=int, default=30)
    parser.add_argument('--stm-to-ltm-threshold', type=int, default=None)
    parser.add_argument('--ltm-max', type=int, default=60)
    parser.add_argument('--window-size', type=int, default=1000)
    parser.add_argument('--novelty-percentile', type=float, default=0.95)
    parser.add_argument('--alpha-stm', type=float, default=0.1)
    parser.add_argument('--ltm-replace-mode', type=str, default='random',
                        choices=['random', 'oldest', 'reservoir', 'similar', 'balanced_time',
                                 'reservoir_oldest', 'reservoir_similar', 'reservoir_balanced_time', 'None'])
    parser.add_argument('--centroid-alpha', type=float, default=0.0)
    parser.add_argument('--trim-instead-of-consolidation', type=str_to_bool, default=False)
    # LUMP params
    parser.add_argument('--alpha-lump', type=float, default=0.4)

    # Buffer Features update with EMA param (originally alpha from minred)
    parser.add_argument('--features-buffer-ema', type=float, default=0.5)

    # CLIMB verbose logging (fifo stats, memory stats, all_embeddings accumulation)
    parser.add_argument('--verbose-climb-log', type=str_to_bool, default=False)

    # Resume training from an existing save_pth (SLURM 20h limit)
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to an existing run folder to resume training from')

    # Deferred evaluation: skip probing during training, save encoder checkpoints, eval later
    parser.add_argument('--deferred-eval', type=str_to_bool, default=False,
                        help='Skip probing during training and run it later via eval_deferred.py')

    args = parser.parse_args()
    
    if args.dataset_seed is None:
        args.dataset_seed = args.seed

    if args.stm_to_ltm_threshold is None:
        args.stm_to_ltm_threshold = args.max_examples_per_centroid

    return args







