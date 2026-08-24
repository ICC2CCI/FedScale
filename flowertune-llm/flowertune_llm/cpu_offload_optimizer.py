"""A deliberately simple DDP optimizer-state CPU offload baseline.

The GPU model remains replicated under DDP. After DDP synchronizes gradients,
this optimizer copies them to FP16 CPU mirror parameters, applies AdamW on the
CPU, then copies the updated parameters back to the GPU. It is meant for a
memory-constrained systems comparison with FSDP, not as a general replacement
for DeepSpeed/ZeRO.
"""

import torch


class CPUOffloadAdamW(torch.optim.Optimizer):
    """AdamW with parameters, gradients, and optimizer state on CPU."""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        # 1e-8 underflows to zero in the FP16 CPU optimizer state and can turn
        # a zero second moment into a divide-by-zero/NaN update after step one.
        eps=1e-4,
        weight_decay=0.01,
    ):
        gpu_params = [param for param in params if param.requires_grad]
        if not gpu_params:
            raise ValueError("CPUOffloadAdamW received no trainable parameters")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(gpu_params, defaults)
        # FP16 mirrors keep OpenLLaMA-3B's CPU optimizer footprint within the
        # 40-GiB GN10X node memory. The experiment records this precision.
        self.cpu_params = [
            torch.nn.Parameter(
                param.detach().to(device="cpu", dtype=torch.float16).clone(),
                requires_grad=True,
            )
            for param in gpu_params
        ]
        self.cpu_optimizer = torch.optim.AdamW(
            self.cpu_params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self._gpu_params = gpu_params

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        # Schedulers update this wrapper's public param_groups. Mirror their
        # hyperparameters into the real CPU optimizer before every update.
        source_group = self.param_groups[0]
        target_group = self.cpu_optimizer.param_groups[0]
        for key in ("lr", "betas", "eps", "weight_decay"):
            target_group[key] = source_group[key]

        for gpu_param, cpu_param in zip(self._gpu_params, self.cpu_params):
            cpu_param.grad = (
                None
                if gpu_param.grad is None
                else gpu_param.grad.detach().to(
                    device="cpu", dtype=torch.float16, copy=True
                )
            )
        self.cpu_optimizer.step()
        for gpu_param, cpu_param in zip(self._gpu_params, self.cpu_params):
            gpu_param.copy_(cpu_param, non_blocking=False)
        return loss

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        self.cpu_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.cpu_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.cpu_optimizer.load_state_dict(state_dict)
