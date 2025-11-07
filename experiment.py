#!/usr/bin/env python3
# model_experiment.py
# conda activate /data2/liym/envs/PLMClassification
# nohup python experiment_dager.py > test11.txt 2>&1 & 

import os
import argparse
import subprocess
import datetime
import json
from pathlib import Path
from itertools import product

def run_experiment(args_dict, experiment_name, base_output_dir):
    """Run a single experiment with the given arguments."""
    # Create command with all arguments
    cmd = ["python", "main.py"]
    for key, value in args_dict.items():
        if value is not None:  # Only add if value is not None
            cmd.extend([f"--{key}", str(value)])
    
    # Print command for logging purposes
    print(f"\n{'='*80}")
    print(f"Running experiment: {experiment_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Create output directory specific to this experiment
    output_dir = os.path.join(base_output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Record experiment settings
    with open(os.path.join(output_dir, "params.json"), "w") as f:
        json.dump(args_dict, f, indent=2)
    
    # Redirect output to log file
    log_file = os.path.join(output_dir, "experiment.log")
    with open(log_file, "w") as f:
        # Run the experiment and capture output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output to both console and log file
        for line in process.stdout:
            print(line, end='')
            f.write(line)
            f.flush()
        
        # Wait for process to complete
        process.wait()
    
    return process.returncode

def main():
    # Timestamp for experiment folder
    timestamp1 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.datetime.now().strftime("%m%d")
    base_output_dir = f"{timestamp}_experiments/dager_{timestamp1}"
    os.makedirs(base_output_dir, exist_ok=True)


    # Base configuration that will be used for all experiments
    base_config = {
        "max_samples": 3000,
        "max_length": 40,
        "num_clients": 4,
        "client_fraction": 1.0,
        "num_rounds": 30, 
        "local_epochs": 1,
        "batch_size": 8,
        "attack_batch_size": 4,
        "lr": 2e-4,
        "avg_lr": 2e-4,
        "seed": 46,
        "adv_weight": 0.8,
        "adv_ratio": 0.5,
        "pgd_steps": 1,
        "eval_pgd_steps":1,# TODO 评估时的PGD步数
        "pgd_alpha": 0.01,
        "gpu_number": 6,
        # dager 参数
        'task': "seq_class",
        "l1_filter": "all",
        "l2_filter": "non-overlap",
        "l1_span_thresh": 0.015,  # 越大越宽松
        "l2_span_thresh": 0.015,  # 越大越宽松
        "rank_tol": 5e-7,     # 越小越宽松，当l1_span_thresh和l2_span_thresh调不好时调这个
        #"l1_span_thresh": 0.25,  # 越大越宽松
        #"l2_span_thresh": 0.10,  # 越大越宽松
        #"rank_tol": 5e-9,     # 越小越宽松，当l1_span_thresh和l2_span_thresh调不好时调这个
        
        'train_method':'lora',
        'target_client':0,
        'precision': 'half',
        'cache_dir':'/data2/liym/data/models_cache/',
        #'cache_dir':'/home/chen/liym/Projects/PRFedLLM/PRFedLLM_V1/dager_experiments/dager_20250730_202433/pgd3_gpt2_medical_abstracts_fedavg/gpt2_medical_abstracts_3clients_fedavg_20250730_203005/final_model',
        'low_bound': 0.01,
        'train_epsilon': 0.04,
        'epsilon': 0.04,
        'lora_r': 256,
        'trimmed_ratio': 0.25
    }
    
    # Define dataset groups
    # datasets = ["pubmed_rct","medical_abstracts"]
    datasets = ["medical_abstracts"]
    
    # Define model sets based on dataset groups
    models_for_datasets = ["gpt2"]
    # Algorithm variations
    # ['fedavg', 'PBLLM','dp-lora2','fat','fedprox,'fedbn','trimmed_mean']
    algorithms=["fedavg"]
    # nohup python experiment_dager.py > test11.txt 2>&1 & 

    # pgd_steps = 1,3,6
    pgd_steps = [ 3]
    #low_bound = 0.01
    
    for pgd_step in pgd_steps:
        # Run experiments for each dataset
        for dataset in datasets:
            print(f"\n{'*'*60}")
            print(f"Dataset: {dataset}")
            print(f"{'*'*60}\n")
            
            # Select the appropriate model set based on the dataset
            models = models_for_datasets
        
            # Run experiments with selected models for this dataset
            for model in models:
                print(f"\n\n{'*'*80}")
                print(f"Starting experiments for model: {model}")
                print(f"{'*'*80}\n")
                
                # Clean model name for directory naming
                model_name_clean = model.replace("/", "_").replace("-", "_")
                
                for algorithm in algorithms:
                    # Basic experiment with current model, dataset and algorithm
                    config = base_config.copy()
                    config["model_path"] = model
                    config["model_name"] = model
                    config["dataset"] = dataset
                    config["algorithm"] = algorithm
                    config["pgd_steps"] = pgd_step
                    config["output_dir"] = os.path.join(base_output_dir, 
                                                    f"pgd{pgd_step}_{model_name_clean}_{dataset}_{algorithm}")
                    # Run the experiment
                    exp_name = f"pgd{pgd_step}_{model_name_clean}_{dataset}_{algorithm}"
                    run_experiment(config, exp_name, base_output_dir)
    
    print("\n\nAll experiments completed!")
    print(f"Results stored in directory: {base_output_dir}")

if __name__ == "__main__":
    main()