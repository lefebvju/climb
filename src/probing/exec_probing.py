import os
import torch
from torch.utils.data import ConcatDataset
from typing import List, Dict

from torch.utils.data import ConcatDataset

from . import AbstractProbe
from ..benchmark import Benchmark


def _already_done(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def exec_probing(kwargs: Dict,
                 probes: List[AbstractProbe],
                 probing_benchmark: Benchmark,
                 encoder: torch.nn,
                 pretr_exp_idx: int,
                 probing_tr_ratio_arr: List[float],
                 save_pth: str,
                 ):
    
    for probe in probes:
        probe_save_pth = probe_save_pth = os.path.join(save_pth, f'probe_{probe.get_name()}')
        print(f'==== Probe {probe.get_name()} ==== ')

    
        # PROBING UTPO (probing on all all experiences up to current pretr_exp_idx)
        if kwargs["probing_upto"]:
            # IID or no SSL training -> generate all upto datasets and probe on each of them
            if kwargs['iid'] or kwargs["no_train"]:
                # Probe upto each experience
                for exp_idx, _ in enumerate(probing_benchmark.train_stream):
                    # Generate upto current exp probing datasets
                    probe_upto_dataset_tr = ConcatDataset([probing_benchmark.train_stream[i] for i in range(exp_idx+1)])
                    probe_upto_dataset_test = ConcatDataset([probing_benchmark.test_stream[i] for i in range(exp_idx+1)])
                    if kwargs['probing_val_ratio'] > 0:
                        probe_upto_dataset_val = ConcatDataset([probing_benchmark.valid_stream[i] for i in range(exp_idx+1)])

                    for probing_tr_ratio in probing_tr_ratio_arr:
                        # Create probing accuracy log file
                        pth = os.path.join(probe_save_pth, f'probing_upto/probing_ratio{probing_tr_ratio}')
                        if not os.path.exists(pth):
                            os.makedirs(pth)
                        probe_save_file = os.path.join(pth, f'probe_exp_{exp_idx}.csv')

                        if _already_done(probe_save_file):
                            print(f'-- Upto Probing (up to exp {exp_idx}), ratio {probing_tr_ratio}: already done, skipping --')
                            continue

                        print(f'-- Upto Probing (up to exp {exp_idx}), probe tr ratio: {probing_tr_ratio} --')

                        if kwargs['probing_val_ratio'] > 0:
                            probe.probe(encoder=encoder, tr_dataset=probe_upto_dataset_tr, test_dataset=probe_upto_dataset_test,
                                        val_dataset=probe_upto_dataset_val, tr_samples_ratio=probing_tr_ratio,
                                        save_file=probe_save_file)
                        else:
                            probe.probe(encoder=encoder, tr_dataset=probe_upto_dataset_tr, test_dataset=probe_upto_dataset_test,
                                        tr_samples_ratio=probing_tr_ratio, save_file=probe_save_file)
            else:
                # Generate upto current exp (pretr_exp_idx) probing datasets
                probe_upto_dataset_tr = ConcatDataset([probing_benchmark.train_stream[i] for i in range(pretr_exp_idx+1)])
                probe_upto_dataset_test = ConcatDataset([probing_benchmark.test_stream[i] for i in range(pretr_exp_idx+1)])
                if kwargs['probing_val_ratio'] > 0:
                    probe_upto_dataset_val = ConcatDataset([probing_benchmark.valid_stream[i] for i in range(pretr_exp_idx+1)])

                for probing_tr_ratio in probing_tr_ratio_arr:
                    # Create probing accuracy log file
                    pth = os.path.join(probe_save_pth, f'probing_upto/probing_ratio{probing_tr_ratio}')
                    if not os.path.exists(pth):
                        os.makedirs(pth)
                    probe_save_file = os.path.join(pth, f'probe_exp_{pretr_exp_idx}.csv')

                    if _already_done(probe_save_file):
                        print(f'-- Upto Probing (up to exp {pretr_exp_idx}), ratio {probing_tr_ratio}: already done, skipping --')
                        continue

                    print(f'-- Upto Probing (up to exp {pretr_exp_idx}), probe tr ratio: {probing_tr_ratio} --')

                    if kwargs['probing_val_ratio'] > 0:
                        probe.probe(encoder=encoder, tr_dataset=probe_upto_dataset_tr, test_dataset=probe_upto_dataset_test,
                                    val_dataset=probe_upto_dataset_val, tr_samples_ratio=probing_tr_ratio,
                                    save_file=probe_save_file)
                    else:
                        probe.probe(encoder=encoder, tr_dataset=probe_upto_dataset_tr, test_dataset=probe_upto_dataset_test,
                                    tr_samples_ratio=probing_tr_ratio, save_file=probe_save_file)
                


        # PROBING SEPARATE
        if kwargs['probing_separate']:
            for probe_exp_idx, probe_tr_exp_dataset in enumerate(probing_benchmark.train_stream):
                probe_test_exp_dataset = probing_benchmark.test_stream[probe_exp_idx]
                if kwargs['probing_val_ratio'] > 0:
                    probe_val_exp_dataset = probing_benchmark.valid_stream[probe_exp_idx]

                # Sample only a portion of the tr samples for probing
                for probing_tr_ratio in probing_tr_ratio_arr:
                    # Create probing accuracy log file
                    pth = os.path.join(probe_save_pth, f'probing_separate/probing_ratio{probing_tr_ratio}')
                    if not os.path.exists(pth):
                        os.makedirs(pth)
                    probe_save_file = os.path.join(pth, f'probe_exp_{pretr_exp_idx}.csv')

                    if _already_done(probe_save_file):
                        print(f'-- Separate Probing on exp {probe_exp_idx}, pretr_exp {pretr_exp_idx}, ratio {probing_tr_ratio}: already done, skipping --')
                        continue

                    print(f'-- Separate Probing on experience: {probe_exp_idx}, probe tr ratio: {probing_tr_ratio} --')
                    if kwargs['probing_val_ratio'] > 0:
                        probe.probe(encoder=encoder, tr_dataset=probe_tr_exp_dataset, test_dataset=probe_test_exp_dataset,
                                    val_dataset=probe_val_exp_dataset, tr_samples_ratio=probing_tr_ratio,
                                    save_file=probe_save_file, exp_idx=probe_exp_idx)
                    else:
                        probe.probe(encoder=encoder, tr_dataset=probe_tr_exp_dataset, test_dataset=probe_test_exp_dataset,
                                    tr_samples_ratio=probing_tr_ratio, save_file=probe_save_file, exp_idx=probe_exp_idx)

        # PROBING JOINT
        if kwargs["probing_joint"]:
            probe_joint_dataset_tr = ConcatDataset(probing_benchmark.train_stream)
            probe_joint_dataset_test = ConcatDataset(probing_benchmark.test_stream)
            if kwargs['probing_val_ratio'] > 0:
                probe_joint_dataset_val = ConcatDataset(probing_benchmark.valid_stream)

            for probing_tr_ratio in probing_tr_ratio_arr:
                # Create probing accuracy log file
                pth = os.path.join(probe_save_pth, f'probing_joint/probing_ratio{probing_tr_ratio}')
                if not os.path.exists(pth):
                    os.makedirs(pth)
                probe_save_file = os.path.join(pth, f'probe_exp_{pretr_exp_idx}.csv')

                if _already_done(probe_save_file):
                    print(f'-- Joint Probing, pretr_exp {pretr_exp_idx}, ratio {probing_tr_ratio}: already done, skipping --')
                    continue

                print(f'-- Joint Probing, probe tr ratio: {probing_tr_ratio} --')

                if kwargs['probing_val_ratio'] > 0:
                    probe.probe(encoder=encoder, tr_dataset=probe_joint_dataset_tr, test_dataset=probe_joint_dataset_test,
                                val_dataset=probe_joint_dataset_val, tr_samples_ratio=probing_tr_ratio,
                                save_file=probe_save_file)
                else:
                    probe.probe(encoder=encoder, tr_dataset=probe_joint_dataset_tr, test_dataset=probe_joint_dataset_test,
                                tr_samples_ratio=probing_tr_ratio, save_file=probe_save_file)