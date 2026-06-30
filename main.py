import uuid
import json

import torch

import os
import datetime
import tqdm as tqdm
import numpy as np
import random
import pandas as pd

from src.get_datasets import get_benchmark, get_iid_dataset, get_downstream_benchmark
from src.probing import exec_probing, ProbingSklearn, ProbingPytorch
from src.backbones import get_encoder

from src.ssl_models import BarlowTwins, SimSiam, BYOL, SimCLR, MAE, recover_ssl_model, STL

from src.strategies import NoStrategy, Replay, CLA_R, CLA_E, CLA_B, LUMP, MinRed, CaSSLe, CaSSLeR
from src.standalone_strategies import SCALE, OsirisR
from src.strategies.climb import CLIMB

from src.trainer import Trainer

from src.buffers import get_buffer

from src.utils import write_final_scores, read_command_line_args, calculate_forgetting, save_avg_stream_acc

import time

from src.logger import get_writer, init_writer
from src.logger import logger
writer=None


def _save_resume_checkpoint(save_pth, exp_idx, ssl_model, trainer, strategy, buffer, strategy_name):
    """Atomically save all state needed to resume after exp_idx."""
    chkpt_dir = os.path.join(save_pth, 'resume_checkpoints')
    os.makedirs(chkpt_dir, exist_ok=True)

    checkpoint = {
        'exp_idx': exp_idx,
        'ssl_model': ssl_model.state_dict(),
        'trainer': trainer.get_resume_state(),
    }

    if strategy_name == 'climb':
        checkpoint['climb_memory'] = strategy.memory
    elif buffer is not None:
        checkpoint['buffer'] = buffer.state_dict()

    tmp = os.path.join(chkpt_dir, 'checkpoint_tmp.pth')
    final = os.path.join(chkpt_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, tmp)
    os.replace(tmp, final)
    print(f'[Resume] Checkpoint saved after exp {exp_idx} → {final}')


def _load_resume_checkpoint(resume_from, ssl_model, trainer, strategy, buffer, strategy_name, device):
    """Load checkpoint and restore all state. Returns last completed exp_idx."""
    chkpt_path = os.path.join(resume_from, 'resume_checkpoints', 'checkpoint_latest.pth')
    if not os.path.exists(chkpt_path):
        raise FileNotFoundError(f'[Resume] No checkpoint found at {chkpt_path}')

    checkpoint = torch.load(chkpt_path, map_location='cpu', weights_only=False)

    ssl_model.load_state_dict(checkpoint['ssl_model'])
    ssl_model.to(device)
    trainer.load_resume_state(checkpoint['trainer'])

    if strategy_name == 'climb' and 'climb_memory' in checkpoint:
        strategy.memory = checkpoint['climb_memory']
        strategy.memory.device = device
    elif 'buffer' in checkpoint and buffer is not None:
        buffer.load_state_dict(checkpoint['buffer'])

    exp_idx = checkpoint['exp_idx']
    print(f'[Resume] Loaded checkpoint: last completed exp={exp_idx}, resuming from exp {exp_idx + 1}')
    return exp_idx


def exec_experiment(**kwargs):
    # Set seed for reproducibility across Python, NumPy, and PyTorch
    seed = kwargs["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    standalone_strategies = ['scale']
    buffer_free_strategies = ['no_strategy', 'cla_b', 'cassle']

    # Checks for CLEAR
    if  kwargs["dataset"] == "clear100":
        if kwargs["num_exps"] != 11:
            print(f'WARNING: Selected number of experiences {kwargs["num_exps"]} is different from default CLEAR100 experiences, resetting to 11 experiences.')
            kwargs["num_exps"] = 11
        if kwargs["iid"]:
            print(f'WARNING: IID pretraining is not supported for CLEAR100, resetting to False.')
            kwargs["iid"] = False

    # Ratios of tr set used for training linear probe
    if kwargs["use_probing_tr_ratios"]:
        probing_tr_ratio_arr = [0.05, 0.1, 0.5, 1]
    else:
        probing_tr_ratio_arr = [1]


    # Set up save folders
    if kwargs.get('resume_from'):
        save_pth = kwargs['resume_from']
        if not os.path.exists(save_pth):
            raise ValueError(f'--resume-from path does not exist: {save_pth}')
        init_writer(save_pth)
        writer = get_writer()
    else:
        unique_id = uuid.uuid4().hex[:6]
        str_now = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")+"_"+unique_id
        if kwargs["strategy"] in standalone_strategies:
                folder_name = f'{kwargs["strategy"]}_{kwargs["dataset"]}_encoder{kwargs["encoder"]}_SEED{kwargs["seed"]}_batch_size{kwargs["repl_mb_size"]+kwargs["tr_mb_size"]}_blurred{kwargs["blurred"]}_numExp{kwargs["num_exps"]}_{str_now}'
        elif kwargs['no_train']:
            folder_name = f'notrain_{kwargs["dataset"]}_encoder{kwargs["encoder"]}_SEED{kwargs["seed"]}_batch_size{kwargs["repl_mb_size"]+kwargs["tr_mb_size"]}_blurred{kwargs["blurred"]}_numExp{kwargs["num_exps"]}_{str_now}'

        else:
            folder_name = f'{kwargs["strategy"]}_{kwargs["model"]}_{kwargs["dataset"]}_encoder{kwargs["encoder"]}_SEED{kwargs["seed"]}_LR{kwargs["lr"]}_OMEGA{kwargs["omega"]}_batch_size{kwargs["repl_mb_size"]}-{kwargs["tr_mb_size"]}_blurred{kwargs["blurred"]}_numExp{kwargs["num_exps"]}_mbPasses{kwargs["mb_passes"]}_mem{kwargs["mem_size"]}_{str_now}'
        if kwargs["iid"]:
            folder_name = f'iid_EPOCH{kwargs["epochs"]}_' + folder_name
        save_pth = os.path.join(kwargs["save_folder"], f'{folder_name}_{kwargs["name"]}')
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)
        init_writer(save_pth)
        writer = get_writer()
        # Always save kwargs so eval_deferred.py can reload them
        with open(os.path.join(save_pth, 'kwargs.json'), 'w') as _f:
            json.dump(kwargs, _f, indent=2)

     # Dataset
    benchmark, image_size, per_exp_classes = get_benchmark(
        dataset_name=kwargs["dataset"],
        dataset_root=kwargs["dataset_root"],
        num_exps=kwargs["num_exps"],
        seed=kwargs["dataset_seed"],
        val_ratio=kwargs["probing_val_ratio"],
        evaluation_protocol_clear=kwargs["evaluation_protocol_clear"],
        blurred=kwargs["blurred"]
        
    )
    if kwargs["iid"]:
        iid_tr_dataset = get_iid_dataset(benchmark)

    # Downstream
    if kwargs["downstream"]:
        downstream_benchmark = get_downstream_benchmark(
            downstream_name=kwargs["downstream_dataset"],
            dataset_root=kwargs["downstream_dataset_root"],
            seed=kwargs["dataset_seed"],
            val_ratio=kwargs["probing_val_ratio"],
        )

    # Save general kwargs (skip on resume — config already written)
    if not kwargs.get('resume_from'):
     with open(save_pth + '/config.txt', 'a') as f:
        f.write('\n')
        f.write(f'---- EXPERIMENT CONFIGS ----\n')
        f.write(f'Seed: {kwargs["seed"]}\n')
        f.write(f'Dataset Seed: {kwargs["dataset_seed"]}\n')
        f.write(f'Experiment Date: {str_now}\n')
        f.write(f'Model: {kwargs["model"]}\n')
        f.write(f'Encoder: {kwargs["encoder"]}\n')
        f.write(f'Dataset: {kwargs["dataset"]}\n')
        if per_exp_classes:
            f.write(f'Per_exp_classes: {per_exp_classes}\n')
        f.write(f'Blurred: {kwargs["blurred"]}\n')
        
        if kwargs["downstream"]:
            f.write(f'Downstream: {kwargs["downstream"]}\n')
            f.write(f'Downstream Dataset: {kwargs["downstream_dataset"]}\n')
        f.write(f'Number of Experiences: {kwargs["num_exps"]}\n')
        f.write(f'Memory Size: {kwargs["mem_size"]}\n')
        f.write(f'MB Passes: {kwargs["mb_passes"]}\n')
        f.write(f'Num Epochs: {kwargs["epochs"]}\n')
        f.write(f'Train MB Size: {kwargs["tr_mb_size"]}\n')
        f.write(f'Replay MB Size: {kwargs["repl_mb_size"]}\n')
        f.write(f'IID pretraining: {kwargs["iid"]}\n')
        f.write(f'Save final model: {kwargs["save_model_final"]}\n')
        f.write(f'-- Pretrained weights initialization configs --\n')
        f.write(f'Pretrain init: {kwargs["pretrain_init_type"]}\n')
        if kwargs["pretrain_init_type"] == 'encoder' or kwargs["pretrain_init_type"] == 'ssl':
            f.write(f'Pretrain init source: {kwargs["pretrain_init_source"]}\n')
            f.write(f'Pretrain init path: {kwargs["pretrain_init_pth"]}\n')

        f.write(f'-- Probing configs --\n')
        f.write(f'Probing after all experiences: {kwargs["probing_all_exp"]}\n')
        f.write(f'Probing on Separated exps: {kwargs["probing_separate"]}\n')
        f.write(f'Probing on Up To current exps: {kwargs["probing_upto"]}\n')
        f.write(f'Probing on all Joint exps: {kwargs["probing_joint"]}\n')
        f.write(f'Probing Validation Ratio: {kwargs["probing_val_ratio"]}\n')
        f.write(f'Probing Train Ratios: {probing_tr_ratio_arr}\n')


   

    

    # Device
    if torch.cuda.is_available():       
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        if kwargs["gpu_idx"] < torch.cuda.device_count():
            device = torch.device(f"cuda:{kwargs['gpu_idx']}")
        else:
            device = torch.device("cuda")
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # Encoder
    encoder, dim_encoder_features = get_encoder(encoder_name=kwargs["encoder"],
                                                image_size=image_size,
                                                ssl_model_name=kwargs["model"],
                                                vit_avg_pooling=kwargs["vit_avg_pooling"],
                                                pretrain_init_type=kwargs["pretrain_init_type"],
                                                pretrain_init_source=kwargs["pretrain_init_source"],
                                                pretrain_init_pth=kwargs["pretrain_init_pth"],
                                                save_pth=save_pth
                                                )



    # Buffer
    buffer = None  # default; overwritten below for strategies that need one
    if not kwargs["strategy"] in buffer_free_strategies:
        if kwargs["buffer_type"] == "default":
            # Set default buffer for each strategy
            if kwargs["strategy"] in ['replay', 'lump', 'double_resnet', 'osiris_r', 'cassle_r']:
                kwargs["buffer_type"] = "reservoir"
            elif kwargs["strategy"] in ['cla_r', 'cla_e']:
                kwargs["buffer_type"] = "fifo"
            elif kwargs["strategy"] == "minred":
                kwargs["buffer_type"] = "minred"
            elif kwargs["strategy"] == "scale":
                kwargs["buffer_type"] = "scale"
            elif kwargs["strategy"] == "climb":
                kwargs["buffer_type"] = "climb"
            else:
                raise Exception(f'Strategy {kwargs["strategy"]} not supported for default buffer')
        # Enforce buffer constraints for certain strategies  
        elif kwargs["buffer_type"] == "scale" and not kwargs["strategy"] == "scale":
            raise Exception(f"Buffer type {kwargs['buffer_type']} is only compatible with strategy 'scale'")
        
        buffer = get_buffer(buffer_type=kwargs["buffer_type"], mem_size=kwargs["mem_size"],
                            alpha_ema=kwargs["features_buffer_ema"],
                            device=device)

        # Save buffer configs
        with open(save_pth + '/config.txt', 'a') as f:
            f.write('\n')
            f.write(f'---- BUFFER CONFIGS ----\n')
            f.write(f'Buffer Type: {kwargs["buffer_type"]}\n')
            f.write(f'Buffer Size: {kwargs["mem_size"]}\n')
            if kwargs["buffer_type"] in ["minred", "reservoir", "fifo"]:
                f.write(f'Features update EMA param (MinRed): {kwargs["features_buffer_ema"]}\n')


    if kwargs["aligner_dim"] <= 0:
        aligner_dim = kwargs["dim_pred"]
    else:
        aligner_dim = kwargs["aligner_dim"]
    
    # ---- SSL model ----
    if not kwargs["strategy"] in standalone_strategies:
        if kwargs["model"] == 'simsiam':
            ssl_model = SimSiam(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                                dim_proj=kwargs["dim_proj"], dim_pred=kwargs["dim_pred"],
                                save_pth=save_pth)
            num_views = 2

        elif kwargs["model"] == 'byol':
            ssl_model = BYOL(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                             dim_proj=kwargs["dim_proj"], dim_pred=kwargs["dim_pred"],
                             byol_momentum=kwargs["byol_momentum"], return_momentum_encoder=kwargs["return_momentum_encoder"],
                             save_pth=save_pth)
            num_views = 2
            
        elif kwargs["model"] == 'barlow_twins':
            ssl_model = BarlowTwins(encoder=encoder, dim_backbone_features=dim_encoder_features,
                                    dim_features=kwargs["dim_proj"],
                                    lambd=kwargs["lambd"], loss_scaling=kwargs["barlow_loss_scaling"], save_pth=save_pth)
            num_views = 2


        elif kwargs["model"] == 'simclr':
            ssl_model = SimCLR(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                             dim_proj=kwargs["dim_proj"], temperature=kwargs["simclr_temp"],
                             save_pth=save_pth)
            num_views = 2
            
        elif kwargs["model"] == 'stl':
            ssl_model = STL(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                             dim_proj=kwargs["dim_proj"], temperature=kwargs["simclr_temp"],
                             save_pth=save_pth)
            num_views = 2

        elif kwargs["model"] == 'osiris_r':
            ssl_model = OsirisR(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                                    dim_proj=kwargs["dim_proj"], buffer=buffer, device=device,
                                    replay_mb_size=kwargs["repl_mb_size"],
                                    save_pth=save_pth)
            num_views = 2
            assert kwargs["strategy"] == kwargs["model"], 'Strategy and SSL model must be the same for Osiris-R'


        elif kwargs["model"] == 'mae':
            ssl_model = MAE(vit_encoder=encoder,
                            image_size=image_size, patch_size=kwargs["mae_patch_size"], emb_dim=kwargs["mae_emb_dim"],
                            decoder_layer=kwargs["mae_decoder_layer"], decoder_head=kwargs["mae_decoder_head"],
                            mask_ratio=kwargs["mae_mask_ratio"], save_pth=save_pth)
            num_views = 1
            
        else:
            raise Exception(f'Invalid model {kwargs["model"]}') 
        
        # Initialization from pretrained weights of SSL model
        if kwargs["pretrain_init_type"] == 'ssl':
            if kwargs["pretrain_init_source"] == 'path':
                ssl_model = recover_ssl_model(ssl_model, kwargs["pretrain_init_pth"])
            else:
                raise Exception(f'Invalid pretrain_init_source for ssl type pretrain initialization: {kwargs["pretrain_init_source"]}')
            
        ssl_model = ssl_model.to(device)
            
    
    # ---- Strategy ----
    if not kwargs["strategy"] in standalone_strategies:
        if kwargs["strategy"] == 'no_strategy':
            strategy = NoStrategy(ssl_model=ssl_model, device=device, save_pth=save_pth)

        elif kwargs["strategy"] == 'replay':
            strategy = Replay(ssl_model=ssl_model, device=device, save_pth=save_pth,
                            buffer=buffer, replay_mb_size=kwargs["repl_mb_size"])
            
        elif kwargs["strategy"] == 'cla_r':
            strategy = CLA_R(ssl_model=ssl_model, device=device, save_pth=save_pth,
                        buffer=buffer, replay_mb_size=kwargs["repl_mb_size"],
                        omega=kwargs["omega"], align_criterion=kwargs["align_criterion"],
                        use_aligner=kwargs["use_aligner"], align_after_proj=kwargs["align_after_proj"], 
                        aligner_dim=aligner_dim)
        
        elif kwargs["strategy"] == 'cla_b':
            strategy = CLA_B(ssl_model=ssl_model, device=device, save_pth=save_pth,
                        omega=kwargs["omega"], align_criterion=kwargs["align_criterion"],
                        use_aligner=kwargs["use_aligner"], align_after_proj=kwargs["align_after_proj"], 
                        aligner_dim=aligner_dim, tau_ema=kwargs["tau_ema"])
        
        elif kwargs["strategy"] == 'cla_e':
            strategy = CLA_E(ssl_model=ssl_model, device=device, save_pth=save_pth,
                            buffer=buffer, replay_mb_size=kwargs["repl_mb_size"],
                            omega=kwargs["omega"], align_criterion=kwargs["align_criterion"],
                            use_aligner=kwargs["use_aligner"], align_after_proj=kwargs["align_after_proj"], 
                            aligner_dim=aligner_dim, tau_ema=kwargs["tau_ema"])
            
        elif kwargs["strategy"] == 'scale':
            pass
            
        elif kwargs["strategy"] == 'lump':
            strategy = LUMP(ssl_model=ssl_model, device=device, save_pth=save_pth,
                            buffer=buffer,
                            alpha_lump=kwargs["alpha_lump"])
            
        elif kwargs["strategy"] == 'minred':
            strategy = MinRed(ssl_model=ssl_model, device=device, save_pth=save_pth,
                            buffer=buffer, replay_mb_size=kwargs["repl_mb_size"])
        
        elif kwargs["strategy"] == 'cassle':
            strategy = CaSSLe(ssl_model=ssl_model, device=device, save_pth=save_pth,
                            omega=kwargs["omega"], align_criterion=kwargs["align_criterion"],
                            use_aligner=kwargs["use_aligner"], align_after_proj=kwargs["align_after_proj"], 
                            aligner_dim=aligner_dim)
            
        elif kwargs["strategy"] == 'cassle_r':
            strategy = CaSSLeR(ssl_model=ssl_model, device=device, save_pth=save_pth,
                            buffer=buffer, replay_mb_size=kwargs["repl_mb_size"],
                            omega=kwargs["omega"], align_criterion=kwargs["align_criterion"],
                            use_aligner=kwargs["use_aligner"], align_after_proj=kwargs["align_after_proj"],
                            aligner_dim=aligner_dim)

        elif kwargs["strategy"] == 'osiris_r':
            strategy = ssl_model # SSL model and strategy are combine

        elif kwargs["strategy"] == 'climb':
            strategy = CLIMB(device=device, file_path=save_pth, ssl_model=ssl_model,
                             dataset_name=kwargs["dataset"], ema=kwargs["ema"],
                             tau_ema=kwargs["tau_ema"],
                             distill_alpha=kwargs["omega"],
                             lr=kwargs["lr"], ltm_merge_strategy=kwargs["ltm_merge_strategy"],
                             update_ltm_bool=kwargs["update_ltm_bool"],
                             bs=kwargs["repl_mb_size"] + kwargs["tr_mb_size"],
                             dim_backbone_features=dim_encoder_features, eta_min=kwargs["eta_min"], ratio_ltm=kwargs["ratio_ltm"], replay_mb_size=kwargs["repl_mb_size"], buffer=buffer, use_aligner=kwargs["use_aligner"], align_after_proj=kwargs["align_after_proj"], aligner_dim=aligner_dim, mem_update=kwargs["pruning"],
                             stm_size=kwargs["stm_size"], max_examples_per_centroid=kwargs["max_examples_per_centroid"], stm_to_ltm_threshold=kwargs["stm_to_ltm_threshold"], ltm_max=kwargs["ltm_max"], window_size=kwargs["window_size"],
                             ltm_replace_mode=kwargs["ltm_replace_mode"],
                             centroid_alpha=kwargs["centroid_alpha"],
                             trim_instead_of_consolidation=kwargs["trim_instead_of_consolidation"],
                             novelty_percentile=kwargs["novelty_percentile"],
                             mem_size=kwargs["mem_size"],
                             alpha=kwargs["alpha_stm"],
                             num_views=num_views,
                             )


        else:
            raise Exception(f'Strategy {kwargs["strategy"]} not supported')

        # Set up the trainer wrapper
        trainer = Trainer(ssl_model=ssl_model, strategy=strategy, optim=kwargs["optim"], lr=kwargs["lr"], momentum=kwargs["optim_momentum"],
                          lars_eta= kwargs["lars_eta"],
                          weight_decay=kwargs["weight_decay"], train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                          mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                          save_model=kwargs["save_model_every_exp"], common_transforms=kwargs["common_transforms"], num_views=num_views)
        
    else:
        if kwargs["strategy"] == 'scale':
            # Is a standalone strategy (already includes trainer and ssl model inside the strategy itself)
            trainer = SCALE(encoder=encoder, optim=kwargs["optim"], lr=kwargs["lr"], dim_backbone_features=dim_encoder_features,
                            momentum=kwargs["optim_momentum"], weight_decay=kwargs["weight_decay"],
                            train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                            mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                            save_model=False, common_transforms=kwargs["common_transforms"],
                            buffer=buffer, replay_mb_size=kwargs["repl_mb_size"],
                            dim_features=kwargs["scale_dim_features"], distill_power=kwargs["scale_distill_power"], buffer_type=kwargs["buffer_type"])

            
            

    # Init probingencoder
    if kwargs["probing_upto"] and not kwargs["probing_all_exp"]:
        raise Exception("Without --probing-all-exp, probing upto is equal to probing joint, please set --probing-upto to false or --probing-all-exp to true")
    
    
    probes = []
    if kwargs["probing_rr"]:
         probes.append(ProbingSklearn(probe_type='rr', device=device, mb_size=kwargs["eval_mb_size"],
                               seed=kwargs["seed"], config_save_pth=save_pth))
    if kwargs["probing_knn"]:
         probes.append(ProbingSklearn(probe_type='knn', device=device, mb_size=kwargs["eval_mb_size"],
                               knn_k=kwargs["knn_k"], seed=kwargs["seed"], config_save_pth=save_pth))
         
    if kwargs["probing_torch"]:
        probes.append(ProbingPytorch(device=device, mb_size=kwargs["eval_mb_size"], config_save_pth=save_pth,
                                 dim_encoder_features=dim_encoder_features, lr=kwargs["probe_lr"],
                                 lr_patience=kwargs["probe_lr_patience"], lr_factor=kwargs["probe_lr_factor"],
                                 lr_min=kwargs["probe_lr_min"], probing_epochs=kwargs["probe_epochs"]))
       

    if kwargs["downstream"]:
        # Using downstream as probing
        probing_benchmark = downstream_benchmark
    else:
        probing_benchmark = benchmark

    training_time_tot = 0 

    if kwargs["iid"]:
        # IID training over the entire dataset
        print(f'==== Beginning self supervised training on iid dataset ====')
        if kwargs["probing_all_exp"]:
            # Evaluate iid trained model during training (not only at the end)
            iid_intermediate_eval_dict = {
                'status': True,
                'num_exps': kwargs["num_exps"],
                'kwargs': kwargs,
                'probes': probes,
                'benchmark': benchmark,
                'probing_tr_ratio_arr': probing_tr_ratio_arr,
            }
        else:
            iid_intermediate_eval_dict = {
                'status': False,
            }

        training_time_start = time.time()
        trained_ssl_model = trainer.train_experience(iid_tr_dataset, exp_idx=0, iid_intermediate_eval_dict=iid_intermediate_eval_dict)
        training_time_tot += time.time() - training_time_start

        if not kwargs["probing_all_exp"]:
            exec_probing(kwargs=kwargs, probes=probes, probing_benchmark=probing_benchmark, encoder=trained_ssl_model.get_encoder_for_eval(), 
                        pretr_exp_idx=0, probing_tr_ratio_arr=probing_tr_ratio_arr, save_pth=save_pth)
        
    elif kwargs["no_train"]:
        # No SSL training is done, only using the randomly initialized encoder as feature extractor
        exec_probing(kwargs=kwargs, probes=probes, probing_benchmark=probing_benchmark, encoder=encoder, pretr_exp_idx=0,
                     probing_tr_ratio_arr=probing_tr_ratio_arr, save_pth=save_pth)

    else:
        # Self supervised training over the experiences
        training_time_start = time.time()

        # Resume: load checkpoint and skip already-completed experiences
        resume_exp_idx = -1
        if kwargs.get('resume_from') and kwargs['strategy'] not in standalone_strategies:
            resume_exp_idx = _load_resume_checkpoint(
                kwargs['resume_from'], ssl_model, trainer, strategy, buffer,
                kwargs['strategy'], device
            )

        for exp_idx, exp_dataset in enumerate(benchmark.train_stream):
            if exp_idx <= resume_exp_idx:
                print(f'[Resume] Skipping exp {exp_idx} (already completed)')
                continue
            if kwargs["blurred"] and exp_idx>0:
                print(f'==== Beginning self supervised training for blurred experience: {exp_idx-1} ====')
                trainer.train_experience(benchmark.blurred_stream[exp_idx-1], exp_idx)
            print(f'==== Beginning self supervised training for experience: {exp_idx} ====')
            trained_ssl_model = trainer.train_experience(exp_dataset, exp_idx)
            training_time_tot += time.time() - training_time_start
            chkpt_pth = os.path.join(save_pth, 'checkpoints')
            if not os.path.exists(chkpt_pth):
                os.makedirs(chkpt_pth)
            torch.save(trained_ssl_model.get_encoder_for_eval().state_dict(),
                        os.path.join(chkpt_pth, str(exp_idx)+'_model_state.pth'))
            # Probing before checkpoint so resume only marks exp done once probing is on disk
            if kwargs["probing_all_exp"] and not kwargs["deferred_eval"]:
                exec_probing(kwargs=kwargs, probes=probes, probing_benchmark=probing_benchmark, encoder=trained_ssl_model.get_encoder_for_eval(),
                     pretr_exp_idx=exp_idx, probing_tr_ratio_arr=probing_tr_ratio_arr, save_pth=save_pth)
            if kwargs['strategy'] not in standalone_strategies:
                _save_resume_checkpoint(save_pth, exp_idx, ssl_model, trainer, strategy,
                                        buffer, kwargs['strategy'])
            
        if not kwargs["probing_all_exp"] and not kwargs["deferred_eval"]:
            # Probe only at the end of training
            exec_probing(kwargs=kwargs, probes=probes, probing_benchmark=probing_benchmark, encoder=trained_ssl_model.get_encoder_for_eval(),
                     pretr_exp_idx=exp_idx, probing_tr_ratio_arr=probing_tr_ratio_arr, save_pth=save_pth)
            
    # Save training time
            with open(os.path.join(save_pth, 'training_time.txt'), 'w') as f:
                f.write(f'Total training time: {training_time_tot} seconds')

        
                
        
    # Calculate and save final probing scores (skipped when deferred_eval=True)
    if kwargs.get("deferred_eval"):
        print('[Deferred eval] Training complete. Run eval_deferred.py to evaluate all checkpoints.')
        return save_pth

    for probe in probes:
        probe_pth = os.path.join(save_pth, f'probe_{probe.get_name()}')
        if kwargs['probing_separate']:
            write_final_scores(probe=probe.get_name(), folder_input_path=os.path.join(probe_pth, 'probing_separate'),
                            output_file=os.path.join(save_pth, 'final_scores_separate.csv'))
        if kwargs['probing_joint']:
            write_final_scores(probe=probe.get_name(), folder_input_path=os.path.join(probe_pth, 'probing_joint'),
                            output_file=os.path.join(save_pth, 'final_scores_joint.csv'))
            if kwargs["probing_all_exp"]:
                save_avg_stream_acc(probe=probe.get_name(), save_pth=save_pth)

        if kwargs['probing_upto'] and not kwargs["probing_joint"]:
            write_final_scores(probe=probe.get_name(), folder_input_path=os.path.join(probe_pth, 'probing_upto'),
                            output_file=os.path.join(save_pth, 'final_scores_joint.csv'))
        #  Calculate forgetting
        if kwargs["probing_separate"] and kwargs["probing_all_exp"] and not (kwargs["iid"] or kwargs["no_train"]):
            calculate_forgetting(save_pth=probe_pth, num_exps=kwargs["num_exps"], probing_tr_ratio_arr=probing_tr_ratio_arr)

        
    # Save final pretrained model
    if kwargs["save_model_final"]:
        chkpt_pth = os.path.join(save_pth, 'checkpoints')
        if not os.path.exists(chkpt_pth):
            os.makedirs(chkpt_pth)
        if kwargs["no_train"]:
            torch.save(encoder.state_dict(),
                    os.path.join(chkpt_pth, f'final_model_state.pth'))
        else:
            if kwargs['strategy'] in standalone_strategies:
                torch.save(trained_ssl_model.get_encoder_for_eval().state_dict(),
                        os.path.join(chkpt_pth, f'final_model_state.pth'))
            else:
                # Default case:
                torch.save(trained_ssl_model.state_dict(),
                        os.path.join(chkpt_pth, f'final_model_state.pth'))


    return save_pth





if __name__ == '__main__':
    # Parse arguments
    args = read_command_line_args()

    exec_experiment(**args.__dict__)
