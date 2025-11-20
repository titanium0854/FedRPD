# -*- coding: utf-8 -*- 
import time
import torch
import numpy as np
import os
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from adversarial import AdversarialAttack
from dager.attack import reconstruct,print_metrics
from dager.utils.models import ModelWrapper
class FederatedServer:
    def __init__(
        self, 
        global_model, 
        clients, 
        test_data, 
        device,
        client_fraction=1.0,
        algorithm='fedavg',
        epsilon=0.05,
        pgd_steps=3,
        eval_pgd_steps=3, # Increased from 1 to 3 for more reliable evaluation
        # Trimmed Mean specific parameters
        trimmed_ratio=0.1,  # Fraction of parameters to trim from each end
    ):
        self.global_model = global_model
        self.clients = clients
        self.test_data = test_data
        self.device = device
        self.client_fraction = client_fraction
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.pgd_steps = pgd_steps
        self.eval_pgd_steps = eval_pgd_steps
        # For FedBN: store the structure of non-BN parameters
        self.non_bn_param_names = None
        self.trimmed_ratio=trimmed_ratio
        
    def select_clients(self):
        """Randomly select a fraction of clients"""
        num_clients = max(1, int(self.client_fraction * len(self.clients)))
        return random.sample(self.clients, num_clients)
    
   
    def aggregate_parameters(self, client_parameters, selected_clients=None):
        """Aggregate client parameters using different methods based on algorithm"""
        if self.algorithm == 'trimmed_mean':
            return self._aggregate_trimmed_mean(client_parameters)
        elif self.algorithm == 'fedbn':
            return self._aggregate_fedbn(client_parameters, selected_clients)
        else:
            return self._aggregate_fedavg(client_parameters)
    
    def _aggregate_fedavg(self, client_parameters):
        """Standard FedAvg aggregation - simple average"""
        # Simple average of client parameters
        aggregated_params = [
            np.mean(
                [client_params[i] for client_params in client_parameters], 
                axis=0
            )
            for i in range(len(client_parameters[0]))
        ]
        
        # Update global model with aggregated parameters
        global_params = self.global_model.state_dict()
        for i, (key, _) in enumerate(global_params.items()):
            global_params[key] = torch.tensor(aggregated_params[i]).to(self.device)
            
        self.global_model.load_state_dict(global_params)
    
    def _aggregate_trimmed_mean(self, client_parameters):
        """Trimmed mean aggregation for robustness against outliers"""
        num_clients = len(client_parameters)
        trim_count = int(num_clients * self.trimmed_ratio)
        
        aggregated_params = []
        
        for i in range(len(client_parameters[0])):
            # Collect the i-th parameter from all clients
            param_values = np.array([client_params[i] for client_params in client_parameters])
            
            # Sort along the client dimension (axis=0)
            sorted_params = np.sort(param_values, axis=0)
            
            # Remove the trimmed portion from both ends
            if trim_count > 0:
                trimmed_params = sorted_params[trim_count:-trim_count]
            else:
                trimmed_params = sorted_params
            
            # Take the mean of the remaining parameters
            aggregated_param = np.mean(trimmed_params, axis=0)
            aggregated_params.append(aggregated_param)
        
        # Update global model with aggregated parameters
        global_params = self.global_model.state_dict()
        for i, (key, _) in enumerate(global_params.items()):
            global_params[key] = torch.tensor(aggregated_params[i]).to(self.device)
            
        self.global_model.load_state_dict(global_params)
        
        print(f"Trimmed Mean: Removed {trim_count} outliers from each end ({num_clients} -> {num_clients - 2*trim_count} clients)")
    
    def _aggregate_fedbn(self, client_parameters, selected_clients):
        """FedBN aggregation - only aggregate non-BN parameters"""
        
        # Get non-BN parameter names from the first client
        if self.non_bn_param_names is None:
            self.non_bn_param_names = selected_clients[0].get_parameter_names()
        
        # Aggregate non-BN parameters using FedAvg
        aggregated_params = [
            np.mean(
                [client_params[i] for client_params in client_parameters], 
                axis=0
            )
            for i in range(len(client_parameters[0]))
        ]
        
        # Update global model with aggregated non-BN parameters only
        global_params = self.global_model.state_dict()
        param_idx = 0
        
        for key in global_params.keys():
            if key in self.non_bn_param_names:
                global_params[key] = torch.tensor(aggregated_params[param_idx]).to(self.device)
                param_idx += 1
            # BN parameters are not updated (keep the current global values)
        
        print("Global model keys:", list(self.global_model.state_dict().keys())[:10])
        print("Client parameter names:", selected_clients[0].get_parameter_names()[:10])

        
        self.global_model.load_state_dict(global_params)
        print(f"FedBN: Aggregated {param_idx} non-BN parameters, kept local BN parameters")

    
    def give_parameters(self, client_parameter, model_wrapper):
        """Aggregate client parameters using FedAvg"""
        # Update global model with aggregated parameters
        global_params = model_wrapper.model.state_dict()
        for i, (key, _) in enumerate(global_params.items()):
            global_params[key] = torch.tensor(client_parameter[i]).to(self.device)
            
        model_wrapper.model.load_state_dict(global_params)
    
    def evaluate(self, adversarial=False):
        """Evaluate the global model on the test data"""
        self.global_model.eval()
        
        test_loader = DataLoader(
            self.test_data, 
            batch_size=8, 
            shuffle=False
        )
        
        correct = 0
        total = 0
        conf_scores = []
        
        # ASR计算: 1. 初始化ASR所需的计数器
        initially_correct_count = 0  # ASR的分母：初始预测正确的样本总数
        successful_attacks_count = 0 # ASR的分子：攻击成功的样本总数

        if adversarial:
            attack = AdversarialAttack(
                self.global_model, 
                epsilon=self.epsilon,
                steps=self.eval_pgd_steps,
                alpha=self.epsilon/(2*self.eval_pgd_steps)
            )
        
        progress_bar = tqdm(test_loader, desc=f"Evaluating {'adversarial (with ASR)' if adversarial else 'normal'}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # --- 对抗评估逻辑 ---
            if adversarial:
                # ASR计算: 2. 首先在干净样本上进行预测，以确定哪些是正确的
                with torch.no_grad():
                    original_outputs = self.global_model(input_ids=input_ids, attention_mask=attention_mask)
                    _, original_predicted = torch.max(original_outputs.logits, 1)
                    
                    # 创建一个mask，标记出那些初始预测正确的样本
                    correct_mask = (original_predicted == labels)
                    num_initially_correct = correct_mask.sum().item()
                    initially_correct_count += num_initially_correct

                # 只对初始预测正确的样本进行攻击，如果没有则跳过攻击步骤
                if num_initially_correct > 0:
                    with torch.enable_grad():
                        # 筛选出需要攻击的数据
                        input_ids_to_attack = input_ids[correct_mask]
                        attention_mask_to_attack = attention_mask[correct_mask]
                        labels_to_attack = labels[correct_mask]
                        
                        # 多轮重启攻击逻辑（保持不变）
                        best_adv_embeds = None
                        worst_correct = float('inf')
                        for _ in range(2):
                            adv_embeds_candidate = attack.generate(input_ids_to_attack, attention_mask_to_attack, labels_to_attack)
                            with torch.no_grad():
                                outputs = self.global_model(inputs_embeds=adv_embeds_candidate, attention_mask=attention_mask_to_attack)
                                probs = torch.softmax(outputs.logits, dim=-1)
                                correct_class_probs = probs[torch.arange(probs.size(0)), labels_to_attack]
                                avg_correct = correct_class_probs.mean().item()
                                if avg_correct < worst_correct:
                                    worst_correct = avg_correct
                                    best_adv_embeds = adv_embeds_candidate.clone()
                        
                        adv_embeds = best_adv_embeds
                    
                    # 获取对抗样本的最终预测结果
                    with torch.no_grad():
                        adv_outputs = self.global_model(inputs_embeds=adv_embeds, attention_mask=attention_mask_to_attack)
                        _, adv_predicted = torch.max(adv_outputs.logits, 1)

                        # ASR计算: 3. 统计攻击成功的数量 (预测结果与真实标签不再相等)
                        successful_attacks = (adv_predicted != labels_to_attack).sum().item()
                        successful_attacks_count += successful_attacks

                        # --- 更新鲁棒准确率的计算（只针对初始正确的样本）---
                        # 鲁棒准确率通常在整个测试集上计算，所以这里我们用所有样本进行一次最终评估
                
                # 为了计算总的鲁棒准确率，我们还是需要对整个batch生成对抗样本并评估
                # （注意：这会重复一些计算，但能确保鲁棒准确率的定义正确）
                with torch.enable_grad():
                    # 重新对整个batch生成最终的对抗样本
                    final_adv_embeds = attack.generate(input_ids, attention_mask, labels)
                
                with torch.no_grad():
                    final_outputs = self.global_model(inputs_embeds=final_adv_embeds, attention_mask=attention_mask)
                    logits = final_outputs.logits
            
            # --- 普通评估逻辑 ---
            else:
                with torch.no_grad():
                    outputs = self.global_model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
            
            probs = torch.softmax(logits, dim=-1)
            values, predicted = torch.max(probs, 1)
            
            conf_scores.extend(values.cpu().tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 在进度条中可以同时显示准确率和ASR
            if adversarial and initially_correct_count > 0:
                current_asr = 100 * successful_attacks_count / initially_correct_count
                progress_bar.set_postfix({
                    "Rob. Acc": f"{100 * correct / total:.2f}%",
                    "ASR": f"{current_asr:.2f}%"
                })
            else:
                progress_bar.set_postfix({"Accuracy": f"{100 * correct / total:.2f}%"})

        # --- 最终结果输出 ---
        accuracy = correct / total
        avg_confidence = np.mean(conf_scores)
        print(f"Test Accuracy ({'Adversarial' if adversarial else 'Normal'}): {accuracy:.4f}, "
            f"Avg Confidence: {avg_confidence:.4f}")

        if adversarial:
            # ASR计算: 4. 计算最终的ASR
            if initially_correct_count > 0:
                asr = successful_attacks_count / initially_correct_count
                print(f"Attack Success Rate (ASR): {asr:.4f}")
            else:
                asr = 0.0
                print("ASR: 0.0 (No samples were initially correct)")
            return accuracy, asr # 返回两个指标

        return accuracy
    
    def train(self, num_rounds, output_dir, metric = None, dager_args = None, model_wrapper = None):
        """Train the global model using federated learning"""
        # Initialize arrays to store metrics
        all_losses = []  # [client_id][round][epoch]
        accuracies = []
        adv_accuracies = []
        round_times = []  # Record time for each round
        
        # Create directory for checkpoints
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for round_num in range(num_rounds):
            print(f"Round {round_num} start: Allocated memory = {torch.cuda.memory_allocated(f'cuda:{dager_args.gpu_number}') / 1024**2:.2f} MB")
            print(f"\n--- Round {round_num+1}/{num_rounds} ---")
            
            # Record start time
            round_start_time = time.time()
            
            # Select clients for this round
            selected_clients = self.select_clients()
            print(f"Selected {len(selected_clients)} clients for training")
            
            # Train on selected clients
            client_parameters = []
            client_losses = [[] for _ in range(len(self.clients))]
            
            true_grad = []
            client_idx = 0
            og_weights = [param.data.clone() for param in self.global_model.parameters()]
            for client in selected_clients:
                # Update client's local model with global model
                client.update_local_model(self.global_model)
                
                losses =[]
                # Perform local training
                losses  = client.train()

                client_losses[client.client_id] = losses
                
                # Collect updated parameters
                client_parameters.append(client.get_parameters())
                
                client_idx += 1

            all_losses.append(client_losses)
            
            # Aggregate parameters
            if self.algorithm == 'fedbn':
                self.aggregate_parameters(client_parameters, selected_clients)
            else:
                self.aggregate_parameters(client_parameters)

            if ((round_num)%5 == 0) or round_num == num_rounds-1:
                # Evaluate global model
                accuracy = self.evaluate(adversarial=False)
                #accuracy =0 
                accuracies.append(accuracy)
                
                # Evaluate with adversarial examples
                adv_accuracy , asr= self.evaluate(adversarial=True)
                #adv_accuracy=0
                adv_accuracies.append(adv_accuracy)
                # Print robustness gap
                rob_gap = accuracy - adv_accuracy
                print(f"Robustness gap: {rob_gap:.4f}")
                print(asr)
                # Calculate round duration
                round_end_time = time.time()
                duration = round_end_time - round_start_time
                round_times.append(duration)
                print(f"Round {round_num+1} duration: {duration:.2f} seconds")
            
            torch.cuda.empty_cache()
            
            # Early stopping if performance is degrading significantly
            if round_num > 5 and self.algorithm in ['cat', 'cat2'] and adv_accuracy < 0.2:
                print("Warning: Adversarial accuracy is very low. Consider reducing epsilon or increasing pgd_steps.")
            print(f"Round {round_num} start: Allocated memory = {torch.cuda.memory_allocated(f'cuda:{dager_args.gpu_number}') / 1024**2:.2f} MB")
        
        # Save results
        self._save_results(output_dir, all_losses, accuracies, adv_accuracies)
        #model_wrapper.tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
        if dager_args.model_path == "gpt2-xl" and (dager_args.train_method == 'lora') and dager_args.finetuned_path is None:
            save_dir = f'./lora_result/gpt2-xl/{dager_args.algorithm}'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'gpt_xl_{dager_args.lora_r}_{dager_args.dataset}.pt')
            torch.save(self.global_model.state_dict(), save_path)
            dager_args.finetuned_path = save_path

        else:
            dager_args.pretrained_model_name_or_path = "./" + output_dir + "/final_model"


        
        # 1) 把“内存里的最终全局模型”直接拷给目标客户端
        import copy

        new_model_wrapper = ModelWrapper(dager_args)
        
        selected_clients[dager_args.target_client].model = copy.deepcopy(new_model_wrapper.model)
        selected_clients[dager_args.target_client].lora_initialized = (self.algorithm == 'dp-lora')
        selected_clients[dager_args.target_client].model.to(self.device)
        # 只对fedavg算法下第一个客户端的最后一轮进行隐私攻击
        if (round_num == num_rounds - 1) and dager_args is not None and model_wrapper is not None:
            # 获取真实梯度，参考dager attack.py里面的compute_grads_fed_avg函数
            #true_grad = [-(param.data.detach() - og_weights[i])/client.batch_size/client.lr/client.local_epochs for i, param in enumerate(client.model.parameters())]
            if self.algorithm == 'FedDualDef':
                true_grad , batch_data = selected_clients[dager_args.target_client].train_FedDualDef()
            elif self.algorithm == 'fedavg'or self.algorithm == 'trimmed_mean':
                true_grad , batch_data = selected_clients[dager_args.target_client].train_avg()
            elif self.algorithm == 'fat':
                true_grad , batch_data = selected_clients[dager_args.target_client].train_fat()
            elif self.algorithm == 'dp-lora':
                true_grad , batch_data = selected_clients[dager_args.target_client].train_dplora()
            elif self.algorithm == 'fedprox':
                true_grad , batch_data = selected_clients[dager_args.target_client].train_fedprox()
            elif self.algorithm == 'fedbn':
                true_grad , batch_data = selected_clients[dager_args.target_client].train_fedbn()
            # 将客户端的参数传递给dager里的模型包装器
            #self.give_parameters(client_parameters[dager_args.target_client], model_wrapper)
            
            # 使用真实梯度进行dager攻击
            prediction, reference = reconstruct(dager_args, batch_data['text'], batch_data['labels'], metric, new_model_wrapper, true_grad)
            
            # 不使用真实梯度攻击
            #prediction, reference = reconstruct(dager_args, batch_data['text'], batch_data['labels'], metric, model_wrapper)
            print('reference: ', flush=True)
            for seq in reference:
                print('========================', flush=True)
                print(seq, flush=True)
            print('========================', flush=True)

            print('predicted: ', flush=True)
            for seq in prediction:
                print('========================', flush=True)
                print(seq, flush=True)
            print('========================', flush=True)
            print('[Curr input metrics]:', flush=True)
            res = metric.compute(predictions=prediction, references=reference, use_aggregator=True)
            print(res)
            with open(os.path.join(output_dir, "results.txt"), "a") as f:
                f.write(f"Attack results:\n")
                f.write('reference: \n')
                for seq in reference:
                    f.write('========================\n')
                    f.write(seq + '\n')
                f.write('========================\n')

                f.write('predicted: \n')
                for seq in prediction:
                    f.write('========================\n')
                    f.write(seq + '\n')
                f.write('========================\n')
                f.write('[Curr input metrics]:\n')
                f.write(str(res) + '\n')
                f.write(f'ASR{asr}')
        # Save round times
        time_file = os.path.join(output_dir, "time.npy")
        np.save(time_file, np.array(round_times))
        print(f"Run times for each round have been saved to {time_file}")
        
    def _save_results(self, output_dir, all_losses, accuracies, adv_accuracies):
        """Save the training results and model"""
        # Save metrics
        np.save(os.path.join(output_dir, "loss.npy"), np.array(all_losses))
        np.save(os.path.join(output_dir, "accuracy.npy"), np.array(accuracies))
        np.save(os.path.join(output_dir, "adv_accuracy.npy"), np.array(adv_accuracies))
        
        # Calculate and save robustness gap
        rob_gap = np.array(accuracies) - np.array(adv_accuracies)
        np.save(os.path.join(output_dir, "robustness_gap.npy"), rob_gap)
        
        # Save model and tokenizer
        self.global_model.save_pretrained(os.path.join(output_dir, "final_model"))
        # Save a text summary
        with open(os.path.join(output_dir, "results.txt"), "w") as f:
            f.write(f"Algorithm: {self.algorithm}\n")
            f.write(f"Final accuracy: {accuracies[-1]:.4f}\n")
            f.write(f"Final adversarial accuracy: {adv_accuracies[-1]:.4f}\n")
            f.write(f"Final robustness gap: {accuracies[-1] - adv_accuracies[-1]:.4f}\n")
            
            # Log accuracy progression
            f.write("\nAccuracy progression:\n")
            for i, acc in enumerate(accuracies):
                f.write(f"Round {i+1}: {acc:.4f}\n")
                
            f.write("\nAdversarial accuracy progression:\n")
            for i, acc in enumerate(adv_accuracies):
                f.write(f"Round {i+1}: {acc:.4f}\n")
                
            f.write("\nRobustness gap progression:\n")
            for i in range(len(accuracies)):
                f.write(f"Round {i+1}: {accuracies[i] - adv_accuracies[i]:.4f}\n")
    
    def computere_grads(model, batch, labels):
        outputs = model(**batch, labels=labels)
        loss = outputs.loss
        loss.backward()
        return model.parameters()