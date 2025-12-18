# FedRPD

## Acknowledgements

This project uses code from [dager-gradient-inversion], developed by INSAIT, Sofia University "St. Kliment Ohridski" and SRI Lab, ETH Zürich. Licensed under the Apache License 2.0. Original source: [https://github.com/insait-institute/dager-gradient-inversion]

## Prerequisites
- Install Anaconda. 
- Create the conda environment:<br>

> conda env create -n FedRPD

- Enable the created environment:<br>

> conda activate FedRPD

- Install requirements.txt

> conda install --file requirements.txt


## Commands
- Running FedRPD with ade_corpus and gpt2
python main.py \
    --dataset ade_corpus \
    --algorithm FedRPD \
    --model_name gpt2 \
    --lora_r 0 \
    --train_method full \
    --precision float32 \
    --l1_span_thresh 0.015 \
    --l2_span_thresh 0.005 \
    --num_clients 4 \
    --num_rounds 30 \
    --local_epochs 1 \
    --batch_size 8 \
    --max_samples 3000 \
    --max_length 40 \
    --lr 0.0002 \
    --seed 42 \
    --target_client 0 \
    --adv_weight 0.65 \
    --adv_ratio 0.5 \
    --pgd_steps 3 \
    --eval_pgd_steps 1 \
    --pgd_alpha 0.01 \
    --train_epsilon 0.04 \
    --epsilon 0.04 \
    --attack_batch_size 4 \
    --low_bound 0.01 \
    --gpu_number 0 \
    --output_dir ./output 

- Running FedRPD with ade_corpus and gpt2-xl
python main.py \
    --dataset ade_corpus \
    --algorithm FedRPD \
    --model_name gpt2-xl \
    --lora_r 256 \
    --train_method lora \
    --precision float32 \
    --l1_span_thresh 0.2 \
    --l2_span_thresh 0.1 \
    --num_clients 4 \
    --num_rounds 30 \
    --local_epochs 1 \
    --batch_size 8 \
    --max_samples 3000 \
    --max_length 40 \
    --lr 0.0002 \
    --seed 42 \
    --target_client 0 \
    --adv_weight 0.65 \
    --adv_ratio 0.5 \
    --pgd_steps 3 \
    --eval_pgd_steps 1 \
    --pgd_alpha 0.01 \
    --train_epsilon 0.04 \
    --epsilon 0.04 \
    --attack_batch_size 4 \
    --low_bound 0.01 \
    --gpu_number 0 \
    --output_dir ./output 

- Running FedRPD with pubmed_rct and gpt2
python main.py \
    --dataset pubmed_rct \
    --algorithm FedRPD \
    --model_name gpt2 \
    --lora_r 0 \
    --train_method full \
    --precision float32 \
    --l1_span_thresh 0.025 \
    --l2_span_thresh 0.005 \
    --num_clients 4 \
    --num_rounds 30 \
    --local_epochs 1 \
    --batch_size 8 \
    --max_samples 3000 \
    --max_length 40 \
    --lr 0.0002 \
    --seed 42 \
    --target_client 0 \
    --adv_weight 0.65 \
    --adv_ratio 0.5 \
    --pgd_steps 3 \
    --eval_pgd_steps 1 \
    --pgd_alpha 0.01 \
    --train_epsilon 0.04 \
    --epsilon 0.04 \
    --attack_batch_size 4 \
    --low_bound 0.01 \
    --gpu_number 0 \
    --output_dir ./output

- Running FedRPD with pubmed_rct and gpt2-xl
python main.py \
    --dataset pubmed_rct \
    --algorithm FedRPD \
    --model_name gpt2-xl \
    --lora_r 256 \
    --train_method lora \
    --precision float32 \
    --l1_span_thresh 0.135 \
    --l2_span_thresh 0.1 \
    --num_clients 4 \
    --num_rounds 30 \
    --local_epochs 1 \
    --batch_size 8 \
    --max_samples 3000 \
    --max_length 40 \
    --lr 0.0002 \
    --seed 42 \
    --target_client 0 \
    --adv_weight 0.65 \
    --adv_ratio 0.5 \
    --pgd_steps 3 \
    --eval_pgd_steps 1 \
    --pgd_alpha 0.01 \
    --train_epsilon 0.04 \
    --epsilon 0.04 \
    --attack_batch_size 2 \
    --low_bound 0.01 \
    --gpu_number 0 \
    --output_dir ./output 

- Running FedRPD with medical_abstracts and gpt2
python main.py \
    --dataset medical_abstracts \
    --algorithm FedRPD \
    --model_name gpt2 \
    --lora_r 0 \
    --train_method full \
    --precision float32 \
    --l1_span_thresh 0.01 \
    --l2_span_thresh 0.01 \
    --num_clients 4 \
    --num_rounds 30 \
    --local_epochs 1 \
    --batch_size 8 \
    --max_samples 3000 \
    --max_length 40 \
    --lr 0.0002 \
    --seed 42 \
    --target_client 0 \
    --adv_weight 0.65 \
    --adv_ratio 0.5 \
    --pgd_steps 3 \
    --eval_pgd_steps 1 \
    --pgd_alpha 0.01 \
    --train_epsilon 0.04 \
    --epsilon 0.04 \
    --attack_batch_size 2 \
    --low_bound 0.01 \
    --gpu_number 0 \
    --output_dir ./output 

- Running FedRPD with medical_abstracts and gpt2-xl
python main.py \
    --dataset medical_abstracts \
    --algorithm FedRPD \
    --model_name gpt2-xl \
    --lora_r 256 \
    --train_method lora \
    --precision float32 \
    --l1_span_thresh 0.25 \
    --l2_span_thresh 0.04 \
    --num_clients 4 \
    --num_rounds 30 \
    --local_epochs 1 \
    --batch_size 8 \
    --max_samples 3000 \
    --max_length 40 \
    --lr 0.0002 \
    --seed 42 \
    --target_client 0 \
    --adv_weight 0.65 \
    --adv_ratio 0.5 \
    --pgd_steps 3 \
    --eval_pgd_steps 1 \
    --pgd_alpha 0.01 \
    --train_epsilon 0.04 \
    --epsilon 0.04 \
    --attack_batch_size 4 \
    --low_bound 0.01 \
    --gpu_number 0 \
    --output_dir ./output 

## Arguments
- `--algorithm`: The federated learning algorithm to execute. Options:
  "PBLLM","fat","fedavg","fedbn","trimmed_mean"