import torch


class PGDAttacker():
    def __init__(self, radius, steps, step_size, random_start, norm_type, lower_bound, ascending=True):
        self.radius = radius * (1) / 255.  #Todo 括号内半径缩放倍数可改，用于调整参数
        # self.radius = radius
        self.steps = steps
        self.step_size = step_size / 255.
        # self.step_size = step_size
        self.random_start = random_start
        self.norm_type = norm_type
        self.ascending = ascending
        self.lower_bound = lower_bound / 255.   # 新增下界

    def perturb(self, model, criterion, input_ids, attention_mask,y):

        x = self.embed(input_ids).detach()

        if self.steps==0 or self.radius==0:
            return x.clone()

        adv_x = x.clone()

        if self.random_start:
            if self.norm_type == 'l-infty':
                adv_x += 2 * (torch.rand_like(x) - 0.5) * self.radius
            else:
                adv_x += 2 * (torch.rand_like(x) - 0.5) * self.radius / self.steps
            adv_x = self._clip_(adv_x, x)

        # if self.random_start:
        #     delta = torch.randn_like(x)
        #     delta_flat = delta.view(delta.shape[0], -1)
        #
        #     if self.norm_type == 'l2':
        #         # 单位球上采样 + 随机缩放到 [lower_bound, radius]
        #         norm = delta_flat.norm(p=2, dim=1, keepdim=True)
        #         unit_delta = delta_flat / (norm + 1e-10)
        #
        #         rand_scales = torch.rand(x.shape[0], 1, device=x.device) * (
        #                     self.radius - self.lower_bound) + self.lower_bound
        #         delta_flat = unit_delta * rand_scales
        #
        #     elif self.norm_type == 'l-infty':
        #         # 均匀采样在 [-radius, radius]
        #         delta_flat = torch.empty_like(delta_flat).uniform_(-self.radius, self.radius)
        #
        #     else:
        #         raise NotImplementedError("random_start only supports l2 and l-infty for now")
        #
        #     delta = delta_flat.view_as(x)
        #     adv_x = x + delta
        #     adv_x = self._clip_(adv_x, x)

        ''' temporarily shutdown autograd of model to improve pgd efficiency '''
        model.eval()
        for pp in model.parameters():
            pp.requires_grad = False

        for step in range(self.steps):
            adv_x.requires_grad_()
            _y = model(inputs_embeds=adv_x, attention_mask=attention_mask)
            loss = criterion(_y, y)
            grad = torch.autograd.grad(loss, [adv_x])[0]

            with torch.no_grad():
                if not self.ascending: grad.mul_(-1)

                if self.norm_type == 'l-infty':
                    adv_x.add_(torch.sign(grad), alpha=self.step_size)
                else:
                    if self.norm_type == 'l2':
                        grad_norm = (grad.reshape(grad.shape[0],-1)**2).sum(dim=1).sqrt()
                    elif self.norm_type == 'l1':
                        grad_norm = grad.reshape(grad.shape[0],-1).abs().sum(dim=1)
                    grad_norm = grad_norm.reshape( -1, *( [1] * (len(x.shape)-1) ) )
                    scaled_grad = grad / (grad_norm + 1e-10)
                    adv_x.add_(scaled_grad, alpha=self.step_size)

                adv_x = self._clip_(adv_x, x)

        ''' reopen autograd of model after pgd '''
        for pp in model.parameters():
            pp.requires_grad = True

        # final_delta = adv_x - x
        # final_norm = final_delta.reshape(final_delta.shape[0], -1).norm(p=2, dim=1)
        # print(f"L2 norm range: min={final_norm.min().item():.4f}, max={final_norm.max().item():.4f}")

        return adv_x.data

    # def _clip_(self, adv_x, x):
    #     delta = adv_x - x  # 计算扰动
    #     # print("Original delta: ", delta)  # 打印原始扰动
    #
    #     if self.norm_type == 'l-infty':
    #         # 限制扰动绝对值在 [lower_bound, radius] 范围内
    #         delta = torch.sign(delta) * delta.abs().clamp(min=self.lower_bound, max=self.radius)
    #     else:
    #         if self.norm_type == 'l2':
    #             norm = (delta.reshape(delta.shape[0], -1)**2).sum(dim=1).sqrt()
    #         elif self.norm_type == 'l1':
    #             norm = delta.reshape(delta.shape[0], -1).abs().sum(dim=1)
    #
    #         norm = norm.reshape(-1, *([1] * (len(x.shape) - 1)))
    #         # 限制扰动的 L2 和 L1 范数，确保它在 [lower_bound, radius] 范围内
    #         norm = norm.clamp(min=self.lower_bound, max=self.radius)
    #         delta *= norm
    #
    #     # print("Clamped delta: ", delta)  # 打印经过限制后的扰动
    #
    #     adv_x = x + delta  # 计算最终的对抗样本
    #     return adv_x

    def _clip_(self, adv_x, x):
        delta = adv_x - x  # 原始扰动

        if self.norm_type == 'l-infty':
            # 对于 L∞，逐元素裁剪并保持在 [lower_bound, radius] 之间
            delta = torch.sign(delta) * delta.abs().clamp(min=self.lower_bound, max=self.radius)

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
            target_norm = current_norm.clamp(min=self.lower_bound, max=self.radius)

            # 按目标范数重构扰动
            delta_flat = unit_delta * target_norm
            delta = delta_flat.view_as(x)

        # 返回合法范围内的 adv_x
        adv_x = x + delta
        return adv_x
