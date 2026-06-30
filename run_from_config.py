import json
import copy
import os

from src.utils import read_command_line_args
from search_hyperparams import search_hyperparams
from multiple_runs import multiple_runs
from main import exec_experiment


# Read args from command line
original_args = read_command_line_args()

print("READING CONFIG ...")


# Read config.json
with open(original_args.config_path) as f:
    config = json.load(f)

    for k, v in config["common_params"].items():
        original_args.__setattr__(k, v)
        # Add also variant with param name with "-" substituted with "_" and vice versa
        original_args.__setattr__(k.replace("_", "-"), v)
        original_args.__setattr__(k.replace("-", "_"), v)

    

    if "experiments" in config:
        log_dir = original_args.save_folder


        for experiment in config["experiments"]:
            if "name" in experiment:
                name = experiment["name"]
            else:
                name = ''

            print(f"Running experiment: {experiment}")
            args = copy.deepcopy(original_args)

            if "hyperparams_search" in experiment:
                # Hyperparameter search

                # Apply experiment specific params
                for k, v in experiment.items():
                    if k not in experiment["hyperparams_search"] and k != "hyperparams_search":
                        args.__setattr__ (k, v)
                        # Add also variant with param name with "-" substituted with "_" and vice versa
                        args.__setattr__(k.replace("_", "-"), v)
                        args.__setattr__(k.replace("-", "_"), v)

                print("experiment.hyperparams:", experiment["hyperparams_search"])
                
                # Run hyperparam search
                search_hyperparams(args, hyperparams_dict=experiment["hyperparams_search"], parent_log_folder=log_dir, experiment_name=name)

            else:
                # Apply experiment specific params
                for k, v in experiment.items():
                    if k != "multiple_runs":
                        args.__setattr__ (k, v)
                        # Add also variant with param name with "-" substituted with "_" and vice versa
                        args.__setattr__(k.replace("_", "-"), v)
                        args.__setattr__(k.replace("-", "_"), v)

                if "multiple_runs" in experiment:
                    # Run multiple runs with different seeds
                    seeds = experiment["multiple_runs"]["seeds"]
                    multiple_runs(args, seeds, parent_log_folder=log_dir, experiment_name=name)
                else:
                    # Single run
                    args.__setattr__('save-folder', log_dir)
                    args.__setattr__('save_folder', log_dir)
                    exec_experiment(**args.__dict__)