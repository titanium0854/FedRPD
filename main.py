# -*- coding: utf-8 -*-
import os
import argparse
import torch
import numpy as np
from datetime import datetime
from server import FederatedServer
from client import FederatedClient
from data_utils import load_and_split_dataset
import random
import json
from dager.utils.models import ModelWrapper

import evaluate
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def get_dager_args(parser):

    parser.add_argument('--neptune', type=str, help='neptune project name, leave empty to not use neptune', default=None)
    parser.add_argument('--neptune_offline', action='store_true', help='Run Neptune in offline mode')
    parser.add_argument('--label', type=str, default='name of the run', required=False)
    
    # Method and setting
    parser.add_argument('--rng_seed', type=int, default=101) 
    #parser.add_argument('--dataset', choices=['cola', 'sst2', 'rte', 'rotten_tomatoes', 'stanfordnlp/imdb', 'glnmario/ECHR'], required=False)
    parser.add_argument('--task', choices=['seq_class', 'next_token_pred'], required=True, default='seq_class')
    parser.add_argument('--pad', choices=['right', 'left'], default='right')
    parser.add_argument('--split', choices=['val', 'test'], default='val', required=False)
    #parser.add_argument('-b','--batch_size', type=int, default=1)
    parser.add_argument('--n_inputs', type=int, default=100, required=False) # val:10/20, test:100
    parser.add_argument('--start_input', type=int, default=0)
    parser.add_argument('--end_input', type=int, default=100000)

    # Model path (defaults to huggingface download, use local path if offline)
    parser.add_argument('--model_path', type=str, default='gpt2')
    parser.add_argument('--finetuned_path', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default='./models_cache')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--device_grad', type=str, default='cpu')
    parser.add_argument('--attn_implementation', type=str, default='sdpa', choices=['sdpa', 'eager'])

    parser.add_argument('--precision', type=str, default='float32', choices=['8bit', 'half', 'float32', 'double'])
    parser.add_argument('--parallel', type=int, default=1000)
    parser.add_argument('--grad_b', type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--rank_tol', type=float, default=5e-07) 
    parser.add_argument('--rank_cutoff', type=int, default=20)
    parser.add_argument('--l1_span_thresh', type=float, default=1e-5) 
    parser.add_argument('--l2_span_thresh', type=float, default=1e-3) 
    parser.add_argument('--l1_filter', choices=['maxB', 'all'], required=True, default='all')
    parser.add_argument('--l2_filter', choices=['overlap', 'non-overlap'], required=True, default='non-overlap')
    parser.add_argument('--distinct_thresh', type=float, default=0.7)
    parser.add_argument('--max_ids', type=int, default=-1)
    parser.add_argument('--maxC', type=int, default=10000000) 
    parser.add_argument('--reduce_incorrect', type=int, default=0)
    parser.add_argument('--n_incorrect', type=int, default=None)
    
    # FedAVG
    parser.add_argument('--algo', type=str, default='fedavg', choices=['sgd', 'fedavg'])
    parser.add_argument('--avg_epochs', type=int, default=None)
    parser.add_argument('--b_mini', type=int, default=None)
    parser.add_argument('--avg_lr', type=float, default=None)
    parser.add_argument('--dist_norm', type=str, default='l2', choices=['l1', 'l2'])
    
    #DP
    parser.add_argument('--defense_noise', type=float, default=None) # add noise to true grads
    #parser.add_argument('--max_len', type=int, default=1e10) 
    parser.add_argument('--p1_std_thrs', type=float, default=5)
    parser.add_argument('--l2_std_thrs', type=float, default=5)
    parser.add_argument('--dp_l2_filter', type=str, default='maxB', choices=['maxB', 'outliers'])
    parser.add_argument('--defense_pct_mask', type=float, default=None) # mask some percentage of gradients
    
    #Dropout
    parser.add_argument('--grad_mode', type=str, default='eval', choices=['eval', 'train'])
    
    #Rebuttal experiments
    parser.add_argument('--hidden_act', type=str, default=None)
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'mse'])
    
    #LoRA
    parser.add_argument('--train_method', type=str, default='lora', choices=['full', 'lora'])
    # parser.add_argument('--lora_r', type=int, default=None)
    parser.add_argument('--label_num', type=int, default='2')


    return parser

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_global_lora_model(model, args):
    # """为全局模型设置LoRA"""
    if args.algorithm == 'dp-lora':
        # 根据模型类型选择target_modules
        model_name = model.__class__.__name__.lower()
        
        # if 'gpt2' in model_name or 'gpt' in model_name:
        #     target_modules = ["c_attn", "c_proj", "c_fc"]
        # elif 'bert' in model_name:
        #     target_modules = ["query", "value", "key", "dense"]
        # elif 'roberta' in model_name:
        #     target_modules = ["query", "value", "key", "dense"]
        # else:
        #     target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]

        target_modules = ["c_attn", "c_proj", "c_fc"]
        
        from peft import LoraConfig, get_peft_model, TaskType
        
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            modules_to_save=["classifier"],
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("[DP-LoRA] Global model initialized with LoRA")
        
    return model

def main():
    parser = argparse.ArgumentParser(description='FedDualDefLLM')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='imdb', choices=['imdb', 'agnews','bbcnews','reuters',
                                                                        'pubmed_rct','medical_abstracts','scotus','ecthr','mimic_attitude','ade_corpus'], 
                        help='Dataset to use')
    parser.add_argument('--max_samples', type=int, default=2000, 
                        help='Maximum number of samples to use per dataset')
    parser.add_argument('--max_length','--max_len', type=int, default=256, 
                        help='Maximum sequence length')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='gpt2-xl', 
                        help='Model name or path,"gpt2')
    
    # Federated learning parameters
    parser.add_argument('--num_clients', type=int, default=3, 
                        help='Number of clients')
    parser.add_argument('--client_fraction', type=float, default=1.0, 
                        help='Fraction of clients to use in each round')
    parser.add_argument('--num_rounds', type=int, default=20, 
                        help='Number of federated rounds')
    parser.add_argument('--local_epochs', type=int, default=1, 
                        help='Number of local epochs per round')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Batch size for client local training')
    parser.add_argument('--lr', type=float, default=2e-4, 
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    
    # Algorithm parameters
    parser.add_argument('--algorithm', type=str, default='fedavg', 
                        choices=['fedavg', 'FedDualDef','fat','fedbn','trimmed_mean'],  # 添加FAT算法
                        help='Federated learning algorithm')
    parser.add_argument('--epsilon', type=float, default=0.5,  # TODO 这个控制扰动的范数大小，取0.02~0.05之间最好吗？
                        help='Epsilon for adversarial training')  
    # parser.add_argument('--noise_epsilon', type=float, default=0.5, 
    #                     help='Epsilon for noise evaluate')
    
    parser.add_argument('--pgd_steps', type=int, default=1,
                        help='PGD steps for adversarial training')
    parser.add_argument('--eval_pgd_steps', type=int, default=1,
                        help='PGD steps for adversarial training')
    
    parser.add_argument('--pgd_alpha', type=float, default=0.01,
                        help='pgd alpha for adversarial training')
    # CAT2 parameters
    parser.add_argument('--adv_weight', type=float, default=0.6,
                        help='Weight for adversarial loss in CAT2')  # TODO 对抗扰动这里，adv_weight 0.8是不是更好一点。
    parser.add_argument('--confidence_threshold', type=float, default=0.9, 
                        help='Confidence threshold for CAT2 algorithm')
    parser.add_argument('--batch_threshold', type=float, default=0.9,
                        help='Batch threshold for CAT2 algorithm')
    
    # FAT parameters
    parser.add_argument('--adv_ratio', type=float, default=0.5,
                        help='Ratio of samples in a batch for adversarial training in FAT')
    
    # Dp-lora噪声 paraments
    parser.add_argument('--dp_sigma', type=float, default=0.0000005,
                        help='Standard deviation of Gaussian noise for DP-LoRA (default: 0.01)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output', 
                        help='Output directory')
    
    parser.add_argument('--low_bound', type=float, default=0.0,
                        help='Lower bound for perturbation, default is 0.0')
    parser.add_argument('--gpu_number',type=int, default=0,
                        help='GPU number to use, default is 0')
    
    # dager parameters
    parser.add_argument('--target_client',type=int, default=0)
    parser.add_argument('--attack_batch_size',type=int, default=0)
    parser = get_dager_args(parser)
    parser.add_argument('--train_epsilon', type=float, default='0.01')

    # FedProx specific parameters
    parser.add_argument('--mu', type=float, default=0.01,
                        help='Proximal term coefficient for FedProx (regularization strength)')

    # FedBN specific parameters
    parser.add_argument('--keep_local_bn', action='store_true', default=True,
                        help='Keep local BatchNorm parameters in FedBN (recommended for non-IID data)')

     # Trimmed Mean specific parameters
    parser.add_argument('--trimmed_ratio', type=float, default=0.1,
                        help='Fraction of outliers to trim from each end in Trimmed Mean aggregation')
    

    # 在parser参数中添加LoRA相关参数
    parser.add_argument('--lora_r', type=int, default=16, 
                    help='LoRA rank parameter')
    parser.add_argument('--lora_alpha', type=int, default=32,
                    help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                    help='LoRA dropout rate')

    args = parser.parse_args()
    
    # Set random seed
    setup_seed(args.seed)
    
    # TODO 这里设置LLM的类别
    dataset_num_labels = {
        'imdb': 2,
        'agnews': 4,
        'bbcnews': 5,
        'reuters': 8,
        'pubmed_rct': 5,       # Medical - PubMed RCT
        'medical_abstracts': 5, # Medical - Medical abstracts by specialty
        'mimic_attitude': 3,  # Medical - mimic_attitude
        'mnli_resampled_as_mednli': 3, # Medical 
        'scotus': 13,        # Legal - Case HOLD
        'ecthr': 8,           # Legal - European Court of Human Rights
        'medical_questions_pairs': 2,
        'ade_corpus': 2
    }

    
    args.label_num = dataset_num_labels.get(args.dataset)
    args.device = f"cuda:{args.gpu_number}"
    args.max_len = args.max_length
    args.model_path = args.model_name 
    print(f"lable_num:{args.label_num},device:{args.device},max_len:{args.max_len}")
    # Load model and tokenizer
    
    device = torch.device(args.device)

    # 评估指标
    metric = evaluate.load('rouge', cache_dir='./dager/models_cache')

    #dataset = TextDataset(args.device, args.dataset, args.split, args.n_inputs, args.batch_size, args.cache_dir)
    # 采用dager的模型包装器中的gpt2作为基础模型
    # 如果要换模型，就修改下面的代码
    model_wrapper = ModelWrapper(args)    
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    model = setup_global_lora_model(model, args)
    device = device
    # Prepare dataset and split for clients
    train_data, test_data = load_and_split_dataset(
        args.dataset, 
        tokenizer, 
        args.max_samples,
        args.max_length,
        args.num_clients
    )
    print(f"train_data length: {[len(train_data[i]) for i in range(len(train_data))]}")
    print(f"test_data length: {len(test_data)}")
    # Initialize clients
    clients = []
    for i in range(args.num_clients):
        clients.append(
            FederatedClient(
                client_id=i,
                train_data=train_data[i],
                model=model,
                tokenizer=tokenizer,
                device=device,
                local_epochs=args.local_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                algorithm=args.algorithm,
                epsilon=args.train_epsilon,
                confidence_threshold=args.confidence_threshold,
                batch_threshold=args.batch_threshold,
                adv_weight=args.adv_weight,
                adv_ratio=args.adv_ratio,  # 添加FAT参数
                pgd_steps=args.pgd_steps,
                pgd_alpha=args.pgd_alpha,
                low_bound=args.low_bound,  # 添加下界
                dp_sigma=args.dp_sigma, # dp-lora噪声参数
                train_method=args.train_method,
                lora_r=args.lora_r,  # 添加LoRA参数
                lora_alpha=args.lora_alpha,  # 添加LoRA参数
                lora_dropout=args.lora_dropout,  # 添加LoRA参数
                attack_batch_size=args.attack_batch_size, 
                mu=args.mu,
                keep_local_bn=args.keep_local_bn,
                
            )
        )
    
    # Initialize server
    server = FederatedServer(
        global_model=model,
        clients=clients,
        test_data=test_data,
        device=device,
        client_fraction=args.client_fraction,
        algorithm=args.algorithm,
        epsilon=args.epsilon,
        # noise_epsilon=args.noise_epsilon  # 添加噪声评估参数
        pgd_steps=args.pgd_steps,
        eval_pgd_steps=args.eval_pgd_steps,
        trimmed_ratio=args.trimmed_ratio,
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        args.output_dir, 
        f"{args.model_name}_{args.dataset}_{args.num_clients}clients_{args.algorithm}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Save hyperparameters
    args_dict = vars(args)
    # 定义将要保存的 JSON 文件路径
    json_path = os.path.join(output_dir, "args.json")

    # 写入 JSON 文件，indent 参数用来格式化输出，使其更容易阅读
    with open(json_path, "w") as f:
        json.dump(args_dict, f, indent=4)


    # Start federated learning
    server.train(args.num_rounds, output_dir, metric, args, model_wrapper)
    
    print(f"Federated learning completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
