import os
import itertools
import datetime
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait
import copy


from main import exec_experiment

def multiple_runs(args, seeds, parent_log_folder='./logs', experiment_name=''):     
     
     process_args_list = []
     for i, seed in enumerate(seeds):
          experiment_args = copy.deepcopy(args)
          experiment_args.__setattr__('seed', seed)
          experiment_args.__setattr__('name', experiment_args.__dict__['name'] + f'_{i}')
          print('<<<<<<<<<<<<<<< Experiment name:', experiment_args.__dict__['name'], '>>>>>>>>>>>>>>>>>')
          print('<<<<<<<<<<<<<<< With seed:', seed, '>>>>>>>>>>>>>>>>>')

          str_now = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
          folder_name = f'multiple_runs_{experiment_name}_{args.strategy}_{args.model}_{str_now}'
          if args.iid:
               folder_name = f'multiple_runs_iid_{experiment_name}_{args.strategy}_{args.model}_{str_now}'
          save_folder = os.path.join(parent_log_folder, folder_name)
          if not os.path.exists(save_folder):
               os.makedirs(save_folder)
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
     final_val_acc_list = []
     final_test_acc_list = []
     avg_val_acc_list = []
     avg_test_acc_list = []
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

          test_acc = results_df['avg_test_acc'].values[0]
          final_test_acc_list.append(test_acc*100)

          if args.probing_val_ratio > 0.0:
               val_acc = results_df['avg_val_acc'].values[0]
               final_val_acc_list.append(val_acc*100)

          if os.path.exists(os.path.join(experiment_save_folder, 'avg_stream_acc.csv')):
               df = pd.read_csv(os.path.join(experiment_save_folder, 'avg_stream_acc.csv'))
               # Select row were column probe_type == probe_type
               df = df[df['probe_type'] == probe_type]
               avg_test_acc_list.append(df['avg_test_acc'].values[0]*100)
               if args.probing_val_ratio > 0.0:
                    avg_val_acc_list.append(df['avg_val_acc'].values[0]*100)

     
     # Write results to file
     with open(os.path.join(save_folder, 'multiple_runs_results.txt'), 'a') as f:
          f.write(f"\n#### Final results: ####\n")
          f.write(f'Final Avg Test Acc List: {final_test_acc_list}\n')
          final_test_mean = sum(final_test_acc_list) / len(final_test_acc_list)
          final_test_std = (sum([(x - final_test_mean) ** 2 for x in final_test_acc_list]) / len(final_test_acc_list)) ** 0.5
          f.write(f"Final Test Acc: {final_test_mean:.3f} +- {final_test_std:.3f}\n")
          f.write(f"Final Test Acc: {final_test_mean:.1f} +- {final_test_std:.1f}\n")
          
          if len(avg_val_acc_list) > 0:
               final_val_mean = sum(final_val_acc_list) / len(final_val_acc_list)
               final_val_std = (sum([(x - final_val_mean) ** 2 for x in final_val_acc_list]) / len(final_val_acc_list)) ** 0.5
               f.write(f'\nFinal Avg Val Acc List: {final_val_acc_list}\n')
               f.write(f"Final Val Acc: {final_val_mean:.3f} +- {final_val_std:.3f}\n")
               f.write(f"Final Val Acc: {final_val_mean:.1f} +- {final_val_std:.1f}\n")

          if len(avg_test_acc_list) > 0:
               f.write(f"\n#### Stream Avg results: ####\n")
               f.write(f'Stream Avg Test Acc List: {avg_test_acc_list}\n')
               avg_test_mean = sum(avg_test_acc_list) / len(avg_test_acc_list)
               avg_test_std = (sum([(x - avg_test_mean) ** 2 for x in avg_test_acc_list]) / len(avg_test_acc_list)) ** 0.5
               f.write(f"Stream Avg Test Acc: {avg_test_mean:.3f} +- {avg_test_std:.3f}\n")
               f.write(f"Stream Avg Test Acc: {avg_test_mean:.1f} +- {avg_test_std:.1f}\n")

          if len(avg_val_acc_list) > 0:
               f.write(f'\nStream Avg Val Acc List: {avg_test_acc_list}\n')
               avg_val_mean = sum(avg_val_acc_list) / len(avg_val_acc_list)
               avg_val_std = (sum([(x - avg_val_mean) ** 2 for x in avg_val_acc_list]) / len(avg_val_acc_list)) ** 0.5
               f.write(f"Stream Avg Val Acc: {avg_val_mean:.3f} +- {avg_val_std:.3f}\n")
               f.write(f"Stream Avg Val Acc: {avg_val_mean:.1f} +- {avg_val_std:.1f}\n")


def launch_exec_experiment(args_dict):
     return exec_experiment(**args_dict)