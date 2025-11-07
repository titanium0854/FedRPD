import torch

class AdversarialAttack:
    """Enhanced PGD attack implementation with stronger attack capabilities"""
    
    def __init__(self, model, epsilon=0.5, steps=3, alpha=None, relative_eps=True, lower_bound=None, norm_type = 'l2'):
        self.model = model
        self.embed = model.get_input_embeddings()
        self.steps = steps
        self.test = 0
        # Fixed alpha option: set a constant step size regardless of steps
        # This prevents the issue where more steps leads to tiny step sizes
        if alpha is None:
            alpha = epsilon / 5  # Fixed fraction of epsilon, not dependent on steps
        self.alpha = alpha
        
        # Calculate relative epsilon based on embedding norm if requested
        if relative_eps:
            with torch.no_grad():
                embed_weights = self.embed.weight
                self.embedding_norm = torch.norm(embed_weights, p=2, dim=-1).mean()
                self.epsilon = epsilon * self.embedding_norm
                print(f"Embedding L2 norm: {self.embedding_norm:.4f}, Effective epsilon: {self.epsilon:.4f}")
                
                if lower_bound is not None:
                    self.lower_bound = lower_bound * self.embedding_norm
                    print(f"Effective lower bound: {self.lower_bound:.4f}")
        else:
            self.epsilon = epsilon
            self.lower_bound = lower_bound
            print(f"Effective epsilon: {self.epsilon:.4f}")
            print(f"Effective lower bound: {self.lower_bound:.4f}")

        # Keep track of attack success rate for debugging
        self.attack_attempts = 0
        self.attack_improvements = 0
        self.norm_type = norm_type
    
    def project_l2(self, original_embeddings, perturbed_embeddings):
        """More stable L2 projection that operates on the difference"""
        # Calculate the perturbation
        delta = perturbed_embeddings - original_embeddings
        
        # Calculate L2 norm along embedding dimension
        norm = torch.norm(delta, p=2, dim=-1, keepdim=True)
        
        # Create a mask where norm exceeds epsilon
        mask = (norm > self.epsilon).squeeze(-1)
        if mask.dim() == 0:  # Handle scalar case
            mask = mask.unsqueeze(0)
            
        # Only project perturbations that exceed epsilon
        if torch.any(mask):
            # Project by scaling down the perturbation
            factor = self.epsilon / norm[mask]
            delta[mask] = delta[mask] * factor
        
        # Return the projected embeddings
        return original_embeddings + delta
    
    def project_l2_v2(self, original_embeddings, perturbed_embeddings):

        # Calculate the perturbation
        delta = perturbed_embeddings - original_embeddings

        # Calculate L2 norm along embedding dimension
        norm = torch.norm(delta, p=2, dim=-1, keepdim=True)

        # 上界mask
        if self.epsilon is not None:
            if self.test == 0:
                self.test = 1
                print("v2 epsilon:", self.epsilon)
            mask_upper = (norm > self.epsilon).squeeze(-1)
            if mask_upper.dim() == 0:
                mask_upper = mask_upper.unsqueeze(0)
            if torch.any(mask_upper):
                # ==========================================================
                #  新增的输出语句在这里
                # ==========================================================
                #num_affected = torch.sum(mask_upper).item()
                #print(f"INFO: 检测到 {num_affected} 个扰动的范数大于上限 {self.epsilon}，正在将其放大。")
                factor = self.epsilon / norm[mask_upper]
                delta[mask_upper] = delta[mask_upper] * factor

        # 下界mask
        if self.lower_bound is not None:
            if self.test == 1:
                self.test = 2
                print("v2 lower_bound:", self.lower_bound)
            mask_lower = (norm < self.lower_bound).squeeze(-1)
            if mask_lower.dim() == 0:
                mask_lower = mask_lower.unsqueeze(0)
            if torch.any(mask_lower):
                # ==========================================================
                #  新增的输出语句在这里
                # ==========================================================
                num_affected = torch.sum(mask_lower).item()
                print(f"INFO: 检测到 {num_affected} 个扰动的范数小于下限 {self.lower_bound}，正在将其放大。")
                # ==========================================================
                factor = self.lower_bound / (norm[mask_lower] + 1e-10)
                delta[mask_lower] = delta[mask_lower] * factor

        return original_embeddings + delta

    def clip(self, x, adv_x):
        delta = adv_x - x  # 原始扰动

        if self.norm_type == 'l-infty':
            # 对于 L∞，逐元素裁剪并保持在 [lower_bound, radius] 之间
            delta = torch.sign(delta) * delta.abs().clamp(min=self.lower_bound, max=self.epsilon)

        else:
            # reshape 成 [B, D] 才能处理每个样本的范数
            delta_flat = delta.reshape(delta.shape[0], -1)

            # 计算当前扰动的范数（L2 或 L1）
            if self.norm_type == 'l2':
                current_norm = delta_flat.norm(p=2, dim=1, keepdim=True)
            elif self.norm_type == 'l1':
                current_norm = delta_flat.norm(p=1, dim=1, keepdim=True)

            # 归一化成单位向量（避免除零）
            unit_delta = delta_flat / (current_norm + 1e-10)

            # 将扰动范数限制在合法区间 [lower_bound, radius]
            target_norm = current_norm.clamp(min=self.lower_bound, max=self.epsilon)

            # 按目标范数重构扰动
            delta_flat = unit_delta * target_norm
            delta = delta_flat.view_as(x)

        # 返回合法范围内的 adv_x
        adv_x = x + delta
        return adv_x
    
    def generate(self, input_ids, attention_mask, labels):
        """Generate adversarial embeddings using PGD with improved attack strength"""
        # Get original embeddings
        with torch.no_grad():
            original_embeddings = self.embed(input_ids).detach()
            
            # Get original loss for comparison
            original_outputs = self.model(
                inputs_embeds=original_embeddings,
                attention_mask=attention_mask,
                labels=labels
            )
            original_loss = original_outputs.loss.item()
        
        # CHANGE 1: Use full epsilon for random initialization (not 0.5)
        noise = torch.randn_like(original_embeddings)
        noise_norm = torch.norm(noise, p=2, dim=-1, keepdim=True)
        noise = noise * self.epsilon / (noise_norm + 1e-10)  # Use full epsilon, not half
        
        # Initialize with random perturbation
        best_embeddings = original_embeddings + noise
        best_embeddings = self.project_l2(original_embeddings, best_embeddings)
        
        # Evaluate initial random perturbation
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=best_embeddings,
                attention_mask=attention_mask,
                labels=labels
            )
            best_loss = outputs.loss.item()
        
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, device=input_ids.device)
        
        # CHANGE 2: Try multiple random restarts for better attack (for evaluation)
        num_restarts = 1
        if self.steps > 5:  # Only use restarts for evaluation, not training
            num_restarts = min(3, self.steps // 5)  # Scale with steps, up to 3
        
        for restart in range(num_restarts):
            # New random initialization for this restart
            if restart > 0:
                noise = torch.randn_like(original_embeddings)
                noise_norm = torch.norm(noise, p=2, dim=-1, keepdim=True)
                noise = noise * self.epsilon / (noise_norm + 1e-10)
                embeddings = original_embeddings + noise
                embeddings = self.project_l2(original_embeddings, embeddings)
            else:
                embeddings = best_embeddings.clone()
            
            # PGD iterations
            for i in range(self.steps):
                # Forward pass with gradient tracking
                embeddings.requires_grad_(True)
                
                # Get loss for maximization (adversarial objective)
                outputs = self.model(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                
                # Compute gradients
                loss.backward()
                
                with torch.no_grad():
                    # Extract gradient
                    grad = embeddings.grad.data
                    
                    # CHANGE 3: Use sign of gradient (true FGSM/PGD) instead of normalized gradient
                    # This typically produces stronger attacks more efficiently
                    grad_sign = torch.sign(grad)
                    
                    # Update embeddings with signed gradient
                    embeddings = embeddings + self.alpha * grad_sign
                    
                    # Project back to epsilon ball
                    embeddings = self.project_l2(original_embeddings, embeddings)
                    
                    # Detach for next iteration
                    embeddings = embeddings.detach()
                
                # Check if this iteration produced a better attack
                with torch.no_grad():
                    outputs = self.model(
                        inputs_embeds=embeddings,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    current_loss = outputs.loss.item()
                    
                    if current_loss > best_loss:
                        best_loss = current_loss
                        best_embeddings = embeddings.clone()
        
        # Track attack improvement statistics
        self.attack_attempts += 1
        if best_loss > original_loss:
            self.attack_improvements += 1
            
        # Periodically print attack success rate
        if self.attack_attempts % 100 == 0:
            success_rate = self.attack_improvements / self.attack_attempts * 100
            print(f"Attack success rate: {success_rate:.2f}% ({self.attack_improvements}/{self.attack_attempts})")
            
        return best_embeddings

    def generate_noise(self, input_ids, attention_mask, labels):
        """
        Generate adversarial embeddings using PGD with improved attack strength
        """
        # Get original embeddings
        with torch.no_grad():
            original_embeddings = self.embed(input_ids).detach()
            
            # Get original loss for comparison
            original_outputs = self.model(
                inputs_embeds=original_embeddings,
                attention_mask=attention_mask,
                labels=labels
            )
            original_loss = original_outputs.loss.item()
        
        # CHANGE 1: Use full epsilon for random initialization (not 0.5)
        noise = torch.randn_like(original_embeddings)
        noise_norm = torch.norm(noise, p=2, dim=-1, keepdim=True)
        noise = noise * self.epsilon / (noise_norm + 1e-10)  # Use full epsilon, not half
        
        # Initialize with random perturbation
        best_embeddings = original_embeddings + noise
        best_embeddings = self.project_l2_v2(original_embeddings, best_embeddings)
        
        # Evaluate initial random perturbation
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=best_embeddings,
                attention_mask=attention_mask,
                labels=labels
            )
            best_loss = outputs.loss.item()
        
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, device=input_ids.device)
        
        # CHANGE 2: Try multiple random restarts for better attack (for evaluation)
        num_restarts = 1
        if self.steps > 5:  # Only use restarts for evaluation, not training
            num_restarts = min(3, self.steps // 5)  # Scale with steps, up to 3
        
        for restart in range(num_restarts):
            # New random initialization for this restart
            if restart > 0:
                noise = torch.randn_like(original_embeddings)
                noise_norm = torch.norm(noise, p=2, dim=-1, keepdim=True)
                noise = noise * self.epsilon / (noise_norm + 1e-10)
                embeddings = original_embeddings + noise
                embeddings = self.project_l2_v2(original_embeddings, embeddings)
            else:
                embeddings = best_embeddings.clone()
            
            # PGD iterations
            for i in range(self.steps):
                # Forward pass with gradient tracking
                embeddings.requires_grad_(True)
                
                # Get loss for maximization (adversarial objective)
                outputs = self.model(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                
                # Compute gradients
                loss.backward()
                
                with torch.no_grad():
                    # Extract gradient
                    grad = embeddings.grad.data
                    
                    # CHANGE 3: Use sign of gradient (true FGSM/PGD) instead of normalized gradient
                    # This typically produces stronger attacks more efficiently
                    grad_sign = torch.sign(grad)
                    
                    # Update embeddings with signed gradient
                    embeddings = embeddings + self.alpha * grad_sign
                    
                    # Project back to epsilon ball
                    embeddings = self.project_l2_v2(original_embeddings, embeddings)
                    
                    # Detach for next iteration
                    embeddings = embeddings.detach()
                
                # Check if this iteration produced a better attack
                with torch.no_grad():
                    outputs = self.model(
                        inputs_embeds=embeddings,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    current_loss = outputs.loss.item()
                    
                    if current_loss > best_loss:
                        best_loss = current_loss
                        best_embeddings = embeddings.clone()
        
        # Track attack improvement statistics
        self.attack_attempts += 1
        if best_loss > original_loss:
            self.attack_improvements += 1
            
        # Periodically print attack success rate
        if self.attack_attempts % 100 == 0:
            success_rate = self.attack_improvements / self.attack_attempts * 100
            print(f"Attack success rate: {success_rate:.2f}% ({self.attack_improvements}/{self.attack_attempts})")
            
        return best_embeddings