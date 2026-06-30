import os
import itertools
import datetime
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait
import copy


from main import exec_experiment

def search_hyperparams(args, hyperparams_dict=None, parent_log_folder='./logs', experiment_name=''):

     standalone_strategies = ['scale']

     if args.probing_val_ratio == 0.0:
          print('WARNING! - probing_val_ratio is 0, cannot execute hyperparams search. Exiting this experiment...')
          return
     
     if hyperparams_dict is None:
          # Define current searched hyperparams in lists
          hyperparams_dict = {
          'lr': [0.1, 0.01, 0.001, 0.0001],
          }
          print('WARNING! - Hyperparams of the experiments not found, using default values:')
          print(hyperparams_dict)
     
     str_now = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")

     if args.strategy in standalone_strategies:
          folder_name = f'hypertune_{experiment_name}_{args.strategy}_{str_now}'
     else:     
          folder_name = f'hypertune_{experiment_name}_{args.strategy}_{args.model}_{str_now}'
     
     if args.iid:
          folder_name = f'hypertune_iid_{experiment_name}_{args.strategy}_{args.model}_{str_now}'
     save_folder = os.path.join(parent_log_folder, folder_name)
     if not os.path.exists(save_folder):
          os.makedirs(save_folder)

     # Save hyperparams
     with open(os.path.join(save_folder, 'hyperparams_config_results.txt'), 'w') as f:
          f.write(str(hyperparams_dict))
          f.write('\n')

     # Get the keys and values from the hyperparameter dictionary
     param_names = list(hyperparams_dict.keys())
     param_values = list(hyperparams_dict.values())

     # Generate all combinations of hyperparameters
     param_combinations = list(itertools.product(*param_values))
     process_args_list = []
     for i, combination in enumerate(param_combinations):
          # process_args_list.append((combination, param_names, save_folder, args, i))

          experiment_args = copy.deepcopy(args)
          param_dict = dict(zip(param_names, combination))
          experiment_args.__setattr__('name', experiment_args.__dict__['name'] + f'_{i}')
          print('<<<<<<<<<<<<<<< Experiment name:', experiment_args.__dict__['name'], '>>>>>>>>>>>>>>>>>')
          print('<<<<<<<<<<<<<<< With params:', param_dict, '>>>>>>>>>>>>>>>>>')

          # Update args with hyperparams
          for k, v in param_dict.items():
               experiment_args.__setattr__(k, v)
               # Add also variant with param name with "-" substituted with "_" and vice versa
               experiment_args.__setattr__(k.replace("_", "-"), v)
               experiment_args.__setattr__(k.replace("-", "_"), v)

          # Set args save_folder
          experiment_args.save_folder = save_folder
          process_args_list.append(experiment_args.__dict__)


     if args.__dict__['max_process'] > 1:
          with ProcessPoolExecutor(max_workers = args.max_process) as executor:
               print(f'Executing hyperparam experiments with multiple processes...')
               save_folders_list = list(executor.map(launch_exec_experiment, process_args_list))
     else:
          print(f'Executing experiments with single process...')
          save_folders_list = [launch_exec_experiment(args) for args in process_args_list]


     # Gather and compare results
     best_val_acc = 0
     probe_type_list = []
     for i, experiment_save_folder in enumerate(save_folders_list):
          # Recover results from experiment
          # Select preferred probing configuration
          probe_config_preferences = ["joint", "upto", "separate"]
          for probing_config in probe_config_preferences:
               if args.__dict__[f'probing_{probing_config}']:
                    results_df = pd.read_csv(os.path.join(experiment_save_folder, f'final_scores_{probing_config}.csv'))
                    break
          # Only row with probe_ratio = 1
          results_df = results_df[results_df['probe_ratio'] == 1]
          # Select preferred probe type
          probe_type_preferences = ["torch", "rr", "knn"]
          for probe_type in probe_type_preferences:
               if probe_type in results_df["probe_type"].to_list():
                    results_df = results_df[results_df["probe_type"] == probe_type]
                    break

          val_acc = results_df['avg_val_acc'].values[0]
          test_acc = results_df['avg_test_acc'].values[0]

          param_dict = dict(zip(param_names, param_combinations[i]))
          with open(os.path.join(save_folder, 'hyperparams_config_results.txt'), 'a') as f:
               f.write(f"{param_dict}, Val Acc: {val_acc}, Test Acc: {test_acc} \n")

          # Compare results
          probe_type_list.append(probe_type)
          if val_acc > best_val_acc:
               best_val_acc = val_acc
               best_test_acc = test_acc
               best_combination = param_dict

     assert all(s == probe_type_list[0] for s in probe_type_list), "Not all experiments have the same preferred probe type, cannot compare results"

     print(f"Best hyperparameter combination found: {best_combination}, with {probe_type} probing")
     # Save to file best combination of hyperparams, test and val accuracies
     with open(os.path.join(save_folder, 'hyperparams_config_results.txt'), 'a') as f:
          f.write(f"\nBest hyperparameter combination {best_combination}, with {probe_type} probing:\n")
          f.write(f"Best Val Acc: {best_val_acc}\n")
          f.write(f"Best Test Acc: {best_test_acc}\n")
          f.write(f'\nTr MB size: {args.tr_mb_size}\n')
          f.write(f'MB passes: {args.mb_passes}\n')

def launch_exec_experiment(args_dict):
     return exec_experiment(**args_dict)