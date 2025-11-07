# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import copy
from adversarial import AdversarialAttack
from tqdm import tqdm
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

class FederatedClient:
    def __init__(
        self, 
        client_id, 
        train_data, 
        model, 
        tokenizer, 
        device, 
        local_epochs=1, 
        batch_size=4, 
        lr=2e-4,
        algorithm='fedavg',
        epsilon=0.05,
        confidence_threshold=0.9,
        batch_threshold=0.7,
        adv_weight=0.6,  # Increased from 0.3 to 0.6 for stronger adversarial influence
        adv_ratio=0.5,   # FAT algorithm parameter
        pgd_steps=3,
        pgd_alpha=0.01,
        low_bound=0.0,
        dp_sigma=0,  # 修正语法错误
        train_method='full',
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        attack_batch_size=1,
        mu=0.01,  # FedProx proximal term coefficient
        # FedBN specific parameters
        keep_local_bn=True,  # Whether to keep local BN parameters
    ):
        self.client_id = client_id
        self.train_data = train_data
        self.device = device
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.confidence_threshold = confidence_threshold
        self.batch_threshold = batch_threshold
        self.adv_weight = adv_weight
        self.adv_ratio = adv_ratio
        self.pgd_steps = pgd_steps
        self.pgd_alpha = pgd_alpha
        

        # DP-lora相关参数
        self.dp_sigma = dp_sigma
        self.train_method = train_method
        self.lora_r = lora_r
        self.lora_alpha=lora_alpha
        self.lora_dropout=lora_dropout

        
        # Create a copy of the model for local training
        self.model = None
        self.base_model = model
        self.tokenizer = tokenizer

        self.lora_initialized = False  # 标记LoRA是否已初始化

        # FedProx parameters
        self.mu = mu

        # FedBN parameters  
        self.keep_local_bn = keep_local_bn
        self.local_bn_params = {}  # Store local BN parameters


       

        #new para
        self.train_loader = DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        self.low_bound = low_bound
        self.attack_batch_size = attack_batch_size

    def update_local_model(self, global_model):
        """更新本地模型，如果不存在则创建"""
        if self.model is None:
            # 初始化一次
            self.model = copy.deepcopy(global_model).to(self.device)
            # Initialize local BN parameters for FedBN
            if self.algorithm == 'fedbn':
                self._initialize_local_bn_params()
        else:
            # Save current local BN parameters before loading global model
            if self.algorithm == 'fedbn':
                self._save_local_bn_params()
            
            # 只加载权重，不重新 deep copy
            self.model.load_state_dict(global_model.state_dict(), strict=False)
            
            # Restore local BN parameters for FedBN
            if self.algorithm == 'fedbn':
                self._restore_local_bn_params()
        
        # Store global model parameters for FedProx
        if self.algorithm == 'fedprox':
            self.global_model_params = {name: param.clone().detach() 
                                      for name, param in global_model.named_parameters()}
        torch.cuda.empty_cache()

    def _initialize_local_bn_params(self):
        """Initialize local BN parameters storage"""
        self.local_bn_params = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                self.local_bn_params[name] = {
                    'weight': module.weight.data.clone() if module.weight is not None else None,
                    'bias': module.bias.data.clone() if module.bias is not None else None,
                    'running_mean': module.running_mean.clone() if hasattr(module, 'running_mean') else None,
                    'running_var': module.running_var.clone() if hasattr(module, 'running_var') else None,
                    'num_batches_tracked': module.num_batches_tracked.clone() if hasattr(module, 'num_batches_tracked') else None
                }
    
    def _save_local_bn_params(self):
        """Save current local BN parameters"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)) and name in self.local_bn_params:
                if module.weight is not None:
                    self.local_bn_params[name]['weight'] = module.weight.data.clone()
                if module.bias is not None:
                    self.local_bn_params[name]['bias'] = module.bias.data.clone()
                if hasattr(module, 'running_mean'):
                    self.local_bn_params[name]['running_mean'] = module.running_mean.clone()
                if hasattr(module, 'running_var'):
                    self.local_bn_params[name]['running_var'] = module.running_var.clone()
                if hasattr(module, 'num_batches_tracked'):
                    self.local_bn_params[name]['num_batches_tracked'] = module.num_batches_tracked.clone()
    
    def _restore_local_bn_params(self):
        """Restore local BN parameters"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)) and name in self.local_bn_params:
                if self.local_bn_params[name]['weight'] is not None:
                    module.weight.data.copy_(self.local_bn_params[name]['weight'])
                if self.local_bn_params[name]['bias'] is not None:
                    module.bias.data.copy_(self.local_bn_params[name]['bias'])
                if self.local_bn_params[name]['running_mean'] is not None:
                    module.running_mean.copy_(self.local_bn_params[name]['running_mean'])
                if self.local_bn_params[name]['running_var'] is not None:
                    module.running_var.copy_(self.local_bn_params[name]['running_var'])
                if self.local_bn_params[name]['num_batches_tracked'] is not None:
                    module.num_batches_tracked.copy_(self.local_bn_params[name]['num_batches_tracked'])
        
   
    # def update_local_model(self, global_model):
    #     """确保客户端和全局模型结构完全一致；如果结构不同，直接用深拷贝替换。"""
    #     if self.model is None:
    #         # 首次初始化：dp-lora 直接复制 LoRA 全局模型；否则复制普通模型
    #         self.model = copy.deepcopy(global_model)
    #         if self.algorithm == 'dp-lora':
    #             self.lora_initialized = True
    #         print(f"[DEBUG] Client {self.client_id} - Local model initialized (dp-lora={self.algorithm=='dp-lora'})")
    #         self.verify_model_structure(global_model)
    #         return

    #     # 后续轮次：优先尝试按 state_dict 加载；若键不对齐则直接替换
    #     g_keys = set(global_model.state_dict().keys())
    #     c_keys = set(self.model.state_dict().keys())
    #     if g_keys == c_keys:
    #         self.model.load_state_dict(global_model.state_dict())
    #     else:
    #         self.model = copy.deepcopy(global_model)
    #         # 如果 algorithm=dp-lora，保持标志一致
    #         if self.algorithm == 'dp-lora':
    #             self.lora_initialized = True
    #         self.verify_model_structure(global_model)


    def verify_model_structure(self, global_model):
        """验证客户端模型和全局模型结构是否一致"""
        client_keys = set(self.model.state_dict().keys())
        global_keys = set(global_model.state_dict().keys())
        
        if client_keys == global_keys:
            print(f"[DEBUG] Client {self.client_id} - Model structure verified: IDENTICAL")
        else:
            missing_in_client = global_keys - client_keys
            extra_in_client = client_keys - global_keys
            
            print(f"[ERROR] Client {self.client_id} - Model structure MISMATCH!")
            print(f"  Missing in client: {len(missing_in_client)}")
            print(f"  Extra in client: {len(extra_in_client)}")
            
            if missing_in_client:
                for key in list(missing_in_client)[:3]:
                    print(f"    Missing: {key}")
            
            if extra_in_client:
                for key in list(extra_in_client)[:3]:
                    print(f"    Extra: {key}")
 

    def train_fat(self): 
        self.model.eval()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        train_loader = DataLoader(
            self.train_data, 
            batch_size=self.attack_batch_size, 
            shuffle=True
        )
        num_steps = 0

        attack = AdversarialAttack(
                self.model, 
                epsilon=self.epsilon, 
                steps=self.pgd_steps,
                alpha=self.pgd_alpha,
                lower_bound=self.low_bound,  # Assuming no lower bound for simplicity
            )
        
        accumulated_grads = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        for batch_idx, batch in enumerate(train_loader):
            return_batch = batch
            for epoch in range(10):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                batch_size = input_ids.size(0)
                K = int(batch_size * self.adv_ratio)  # Calculate number of adversarial samples
                
                optimizer.zero_grad()
                
                if K > 0:
                    # Generate adversarial embeddings for first K samples
                    adv_embeddings = attack.generate(
                        input_ids[:K], 
                        attention_mask[:K], 
                        labels[:K]
                    )
                    
                    # Get original embeddings for remaining samples
                    with torch.no_grad():
                        orig_embeddings = self.model.get_input_embeddings()(input_ids[K:])
                    
                    # Combine adversarial and original embeddings
                    combined_embeddings = torch.cat([adv_embeddings, orig_embeddings], dim=0)
                    
                    # Forward pass and loss calculation
                    outputs = self.model(
                        inputs_embeds=combined_embeddings,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                else:
                    # Standard training if K=0
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                
                loss.backward()
                
                # Apply gradient clipping
                if self.train_method  == 'full':
                    for i, param in enumerate(self.model.parameters()):
                        if param.requires_grad and param.grad is not None:
                            accumulated_grads[i] += param.grad.clone()
                else:
                    grad_index = 0
                    for param in self.model.parameters():
                        if param.requires_grad and param.grad is not None:
                            accumulated_grads[grad_index] += param.grad.clone()
                            grad_index += 1
                #optimizer.step()
                
                num_steps += 1
            break
        if num_steps > 0:
            captured_grads = [grad / num_steps for grad in accumulated_grads]
        else:
            captured_grads = None # 如果没有训练步骤，则没有梯度
        
        self.model.train()
        return captured_grads, return_batch
    

    def train_PRFed(self):
        self.model.eval()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        train_loader = DataLoader(
            self.train_data, 
            batch_size=self.attack_batch_size, 
            shuffle=True
        )
        num_steps = 0

        attack = AdversarialAttack(
                self.model, 
                epsilon=self.epsilon, 
                steps=self.pgd_steps,
                alpha=self.pgd_alpha,
               # relative_eps=False,
                lower_bound=self.low_bound,  # Assuming no lower bound for simplicity
            )
        
        accumulated_grads = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        for batch_idx, batch in enumerate(train_loader):
            return_batch = batch
            for epoch in range(10):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                #labels = labels.unsqueeze(0)  # Flatten labels for sequence classification

                # Decide whether to use adversarial training for CAT2
                use_adversarial = True
                                
                # 1) Standard forward pass and loss
                normal_out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                normal_loss = normal_out.loss

                # 2) Adversarial loss calculation (if applicable)
                adv_loss = 0.0
                if use_adversarial:
                    # Generate adversarial embeddings
                    adv_embeds = attack.generate_noise(input_ids, attention_mask, labels)
                    
                    # Compute adversarial loss
                    adv_out = self.model(
                        inputs_embeds=adv_embeds,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    adv_loss = adv_out.loss
                    
                    # Add KL divergence regularization to maintain decision boundary
                    with torch.no_grad():
                        normal_probs = torch.softmax(normal_out.logits, dim=-1)
                    
                    adv_probs = torch.softmax(adv_out.logits, dim=-1)
                    kl_div = nn.KLDivLoss(reduction='batchmean')(
                        (adv_probs + 1e-8).log(), 
                        (normal_probs + 1e-8)
                    )
                    adv_loss += 0.5 * kl_div

                # 3) Combine losses with adaptive weighting
                # Weight adversarial loss higher when model is more confident
                if use_adversarial:
                    # Adaptive weighting: increase adversarial weight as training progresses
                    #current_adv_weight = min(self.adv_weight * (1.0 + epoch/self.local_epochs), 0.7)
                    current_adv_weight = self.adv_weight
                    normal_weight = 1.0 - current_adv_weight
                    loss = normal_weight * normal_loss + current_adv_weight * adv_loss
                else:
                    loss = normal_loss

                optimizer.zero_grad()
                loss.backward()
                
                # Apply gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if self.train_method  == 'full':
                    for i, param in enumerate(self.model.parameters()):
                        if param.requires_grad and param.grad is not None:
                            accumulated_grads[i] += param.grad.clone()
                else:
                    grad_index = 0
                    for param in self.model.parameters():
                        if param.requires_grad and param.grad is not None:
                            accumulated_grads[grad_index] += param.grad.clone()
                            grad_index += 1

                #optimizer.step()
                
                num_steps += 1
            break
            
        if num_steps > 0:
            captured_grads = [grad / num_steps for grad in accumulated_grads]
        else:
            captured_grads = None # 如果没有训练步骤，则没有梯度
        
        self.model.train()
        return captured_grads,return_batch
    

    def train_avg(self):
        self.model.eval()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        train_loader = DataLoader(
            self.train_data, 
            batch_size=self.attack_batch_size, 
            shuffle=True
        )
        num_steps = 0
        accumulated_grads = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        for batch_idx, batch in enumerate(train_loader):
            return_batch = batch
            for epoch in range(5):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                batch_size = input_ids.size(0)
                K = int(batch_size * self.adv_ratio)  # Calculate number of adversarial samples
                
                optimizer.zero_grad()
                # Standard training if K=0
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                
                loss.backward()
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # optimizer.step()
                # Apply gradient clipping
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                 # 累加梯度
                grad_index = 0
                for param in self.model.parameters():
                    if param.requires_grad and param.grad is not None:
                        accumulated_grads[grad_index] += param.grad.clone()
                        grad_index += 1
                                
                num_steps += 1
                #只从train_loader中随机挑选一个batch_size大小的数据进行训练
            break
        if num_steps > 0:
            captured_grads = [grad / num_steps for grad in accumulated_grads]
        else:
            captured_grads = None # 如果没有训练步骤，则没有梯度
        
        self.model.train()
        return captured_grads,return_batch

    
    
    def train_fedprox(self):
        """FedProx：只捕获梯度（不执行 optimizer.step）,返回 averaged grads 和 一个 batch"""
        self.model.eval()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)  # 用于计算梯度（不 step）
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.attack_batch_size,
            shuffle=True
        )

        num_steps = 0
        # 为所有需要梯度的参数准备累加容器（顺序与 model.parameters() 对应）
        grads_template = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]

        for batch_idx, batch in enumerate(train_loader):
            return_batch = batch
            # 与 train_avg 一致，这里用固定的 epoch 次数（train_avg 使用 5）
            for epoch in range(5):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }

                optimizer.zero_grad()
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss

                # 计算 FedProx proximal loss（如果 global_model_params 存在）
                prox_loss = self._compute_proximal_loss()
                total_loss = loss + prox_loss

                # 反向传播以计算梯度（但不执行 optimizer.step()）
                total_loss.backward()

                # 梯度裁剪以保持稳定
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # 累积梯度（与 model.parameters() 的顺序一致）
                grad_idx = 0
                for param in self.model.parameters():
                    if param.requires_grad and param.grad is not None:
                        grads_template[grad_idx] += param.grad.detach().clone()
                        grad_idx += 1

                num_steps += 1

            # 只采样一个 batch 后退出（与 train_avg 行为一致）
            break

        # 计算平均梯度
        if num_steps > 0:
            captured_grads = [g / num_steps for g in grads_template]
        else:
            captured_grads = None

        self.model.train()
        return captured_grads, return_batch
    
    def train_fedbn(self):
        """FedBN：只捕获梯度（不执行 optimizer.step），返回 averaged grads 和 一个 batch"""
        self.model.eval()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)  # 仅用于计算梯度（不执行 step）
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.attack_batch_size,
            shuffle=True
        )

        num_steps = 0
        grads_template = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]

        for batch_idx, batch in enumerate(train_loader):
            return_batch = batch
            # 与 train_fedprox 保持一致，固定训练 5 轮
            for epoch in range(5):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }

                optimizer.zero_grad()
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss

                # 反向传播计算梯度（不执行 optimizer.step）
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # 累积梯度
                grad_idx = 0
                for param in self.model.parameters():
                    if param.requires_grad and param.grad is not None:
                        grads_template[grad_idx] += param.grad.detach().clone()
                        grad_idx += 1

                num_steps += 1

            # 只采样一个 batch 后退出
            break

        # 计算平均梯度
        if num_steps > 0:
            captured_grads = [g / num_steps for g in grads_template]
        else:
            captured_grads = None

        self.model.train()
        return captured_grads, return_batch



    

    def train(self):
        self.model.train()

        # Use different learning rates for different algorithms
        if self.algorithm in ['cat', 'cat2']:
            lr = self.lr * 0.8
        elif self.algorithm == 'fedprox':
            lr = self.lr * 0.9  # Slightly lower LR for FedProx stability
        else:
            lr = self.lr
        # Use gradient clipping for improved stability
        optimizer = AdamW(self.model.parameters(), lr=lr)
        
        train_loader = DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        epoch_losses = []
        
        if self.algorithm == 'fedavg' or self.algorithm == 'trimmed_mean':
            for epoch in range(self.local_epochs):
                total_loss = 0
                progress_bar = tqdm(train_loader, desc=f"Client {self.client_id} - Epoch {epoch+1}/{self.local_epochs}")
                for batch in progress_bar:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    batch_size = input_ids.size(0)
                    K = int(batch_size * self.adv_ratio)  # Calculate number of adversarial samples
                    
                    optimizer.zero_grad()
                    # Standard training if K=0
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
                epoch_loss = total_loss / len(train_loader)
                epoch_losses.append(epoch_loss)
                print(f"Client {self.client_id} - Epoch {epoch+1}/{self.local_epochs} | Avg Loss: {epoch_loss:.4f}")

        
        elif self.algorithm == 'fat':
            # FAT implementation with improved adversarial attack
            attack = AdversarialAttack(
                self.model, 
                epsilon=self.epsilon,
                steps=self.pgd_steps,
                alpha=self.pgd_alpha
            )
            
            for epoch in range(self.local_epochs):
                total_loss = 0
                progress_bar = tqdm(train_loader, desc=f"Client {self.client_id} - Epoch {epoch+1}/{self.local_epochs}")
                for batch in progress_bar:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    batch_size = input_ids.size(0)
                    K = int(batch_size * self.adv_ratio)  # Calculate number of adversarial samples
                    
                    optimizer.zero_grad()
                    
                    if K > 0:
                        # Generate adversarial embeddings for first K samples
                        adv_embeddings = attack.generate(
                            input_ids[:K], 
                            attention_mask[:K], 
                            labels[:K]
                        )
                        
                        # Get original embeddings for remaining samples
                        with torch.no_grad():
                            orig_embeddings = self.model.get_input_embeddings()(input_ids[K:])
                        
                        # Combine adversarial and original embeddings
                        combined_embeddings = torch.cat([adv_embeddings, orig_embeddings], dim=0)
                        
                        # Forward pass and loss calculation
                        outputs = self.model(
                            inputs_embeds=combined_embeddings,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                    else:
                        # Standard training if K=0
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                    
                    loss.backward()
                    
                    # Apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
                epoch_loss = total_loss / len(train_loader)
                epoch_losses.append(epoch_loss)
                print(f"Client {self.client_id} - Epoch {epoch+1}/{self.local_epochs} | Avg Loss: {epoch_loss:.4f}")
        
        elif self.algorithm in ['cat', 'cat2', 'PBLLM']:
            # Enhanced CAT/CAT2 implementation
            attack = AdversarialAttack(
                self.model, 
                epsilon=self.epsilon, 
                steps=self.pgd_steps,
                alpha=self.pgd_alpha,
                lower_bound=self.low_bound,  # Assuming no lower bound for simplicity
            )

            for epoch in range(self.local_epochs):
                total_loss = 0
                adv_batches = 0  # Track how many batches used adversarial training
                total_batches = 0
                progress_bar = tqdm(train_loader, desc=f"Client {self.client_id} - Epoch {epoch+1}/{self.local_epochs}")
                
                for batch in progress_bar:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    total_batches += 1

                    # Decide whether to use adversarial training for CAT2
                    use_adversarial = True
                    if self.algorithm == 'cat2':
                        with torch.no_grad():
                            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                            probs = torch.softmax(logits, dim=-1)
                            confidences, _ = torch.max(probs, dim=1)
                            high_conf_ratio = (confidences >= self.confidence_threshold).float().mean().item()
                            use_adversarial = high_conf_ratio < self.batch_threshold
                    
                    if use_adversarial:
                        adv_batches += 1
                    
                    # 1) Standard forward pass and loss
                    normal_out = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    normal_loss = normal_out.loss

                    # 2) Adversarial loss calculation (if applicable)
                    adv_loss = 0.0
                    if use_adversarial:
                        # Generate adversarial embeddings
                        if self.algorithm in ['cat', 'cat2']:
                            adv_embeds = attack.generate(input_ids, attention_mask, labels)
                        else:  # PBLLM
                            adv_embeds = attack.generate_noise(input_ids, attention_mask, labels)
                        # Compute adversarial loss
                        adv_out = self.model(
                            inputs_embeds=adv_embeds,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        adv_loss = adv_out.loss
                        
                        # Add KL divergence regularization to maintain decision boundary
                        with torch.no_grad():
                            normal_probs = torch.softmax(normal_out.logits, dim=-1)
                        
                        adv_probs = torch.softmax(adv_out.logits, dim=-1)
                        kl_div = nn.KLDivLoss(reduction='batchmean')(
                            (adv_probs + 1e-8).log(), 
                            (normal_probs + 1e-8)
                        )
                        adv_loss += 0.5 * kl_div

                    # 3) Combine losses with adaptive weighting
                    # Weight adversarial loss higher when model is more confident
                    if use_adversarial:
                        # Adaptive weighting: increase adversarial weight as training progresses
                        #current_adv_weight = min(self.adv_weight * (1.0 + epoch/self.local_epochs), 0.7)
                        current_adv_weight = self.adv_weight
                        normal_weight = 1.0 - current_adv_weight
                        loss = normal_weight * normal_loss + current_adv_weight * adv_loss
                    else:
                        loss = normal_loss

                    # 4) Backpropagation and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Apply gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()

                    total_loss += loss.item()
                    progress_bar.set_postfix({
                        "Loss": f"{loss.item():.4f}", 
                        "Adv": f"{use_adversarial}"
                    })

                # Print epoch statistics
                epoch_loss = total_loss / len(train_loader)
                epoch_losses.append(epoch_loss)
                adv_ratio = adv_batches / total_batches if total_batches > 0 else 0
                print(f"Client {self.client_id} - Epoch {epoch+1}/{self.local_epochs} | "
                      f"Avg Loss: {epoch_loss:.4f} | Adv Batches: {adv_ratio:.2f}")

        elif self.algorithm == 'fedprox':
            # FedProx training with proximal term
            epoch_losses = self._train_fedprox(optimizer, train_loader)
        
        elif self.algorithm == 'fedbn':
            # FedBN training (same as FedAvg but BN handling is different)
            epoch_losses = self._train_fedbn(optimizer, train_loader) 
        

        return epoch_losses

    def _train_fedbn(self, optimizer, train_loader):
        """FedBN training (same as FedAvg but with BN parameter management)"""
        epoch_losses = []
        
        for epoch in range(self.local_epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Client {self.client_id} - FedBN - Epoch {epoch+1}/{self.local_epochs}")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }

                optimizer.zero_grad()
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
            epoch_loss = total_loss / len(train_loader)
            epoch_losses.append(epoch_loss)
            print(f"Client {self.client_id} - FedBN - Epoch {epoch+1}/{self.local_epochs} | Avg Loss: {epoch_loss:.4f}")
        
        return epoch_losses
    
    def _train_fedprox(self, optimizer, train_loader):
        """FedProx training with proximal term"""
        epoch_losses = []
        
        for epoch in range(self.local_epochs):
            total_loss = 0
            total_prox_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Client {self.client_id} - FedProx - Epoch {epoch+1}/{self.local_epochs}")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }

                optimizer.zero_grad()
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss

                
                # Add proximal term
                prox_loss = self._compute_proximal_loss()
                total_loss_with_prox = loss + prox_loss
                total_loss_with_prox.backward()
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                total_prox_loss += prox_loss.item() if isinstance(prox_loss, torch.Tensor) else prox_loss
                progress_bar.set_postfix({
                    "Loss": f"{loss.item():.4f}", 
                    "Prox": f"{prox_loss.item() if isinstance(prox_loss, torch.Tensor) else prox_loss:.4f}"
                })
            
            epoch_loss = total_loss / len(train_loader)
            epoch_prox_loss = total_prox_loss / len(train_loader)
            epoch_losses.append(epoch_loss)
            print(f"Client {self.client_id} - FedProx - Epoch {epoch+1}/{self.local_epochs} | "
                  f"Avg Loss: {epoch_loss:.4f} | Prox Loss: {epoch_prox_loss:.4f}")
        
        return epoch_losses
    

    def get_parameters(self):
        """返回模型参数，但不删除模型"""
        if self.model is None:
            raise RuntimeError(f"Client {self.client_id}: No model available. Call update_local_model() first.")
        
        # For FedBN, exclude BN parameters from aggregation
        if self.algorithm == 'fedbn':
            parameters = []
            for name, param in self.model.named_parameters():
                # Skip BN/LayerNorm parameters
                is_bn_param = False
                for module_name, module in self.model.named_modules():
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                        if name.startswith(module_name):
                            is_bn_param = True
                            break
                
                if not is_bn_param:
                    parameters.append(param.cpu().detach().to(torch.float).numpy())
            return parameters
        else:
            # Standard parameter extraction
            parameters = [val.cpu().detach().to(torch.float).numpy() for _, val in self.model.state_dict().items()]
            return parameters
    
    def get_parameter_names(self):
        """Get parameter names (useful for FedBN)"""
        if self.model is None:
            raise RuntimeError(f"Client {self.client_id}: No model available.")
        
        if self.algorithm == 'fedbn':
            param_names = []
            for name, param in self.model.named_parameters():
                # Skip BN/LayerNorm parameters
                is_bn_param = False
                for module_name, module in self.model.named_modules():
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                        if name.startswith(module_name):
                            is_bn_param = True
                            break
                
                if not is_bn_param:
                    param_names.append(name)
            return param_names
        else:
            return list(self.model.state_dict().keys())
    
    def _compute_proximal_loss(self):
        """Compute FedProx proximal term"""
        if self.global_model_params is None:
            return 0.0
        
        proximal_loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.global_model_params:
                proximal_loss += torch.norm(param - self.global_model_params[name].to(self.device)) ** 2
        
        return self.mu / 2.0 * proximal_loss
        

    # def get_parameters(self):
    #     """返回模型参数"""
    #     if self.algorithm == 'dp-lora':
    #         # 返回LoRA参数字典
    #         return self.get_lora_parameters()
    #     else:
    #         return [val.cpu().detach().to(torch.float).numpy() 
    #                for _, val in self.model.state_dict().items()]
    
   
    # 保持其他方法不变...
    def get_copied_model(self):
        return copy.deepcopy(self.model)
    
    def frozen_net(self, frozen):
        for param in self.model.parameters():
            param.requires_grad = not frozen
        if frozen:
            self.model.eval()
        else:
            self.model.train()