import gc
import time
import torch

def _stateless_init_process_group(master_address, master_port, rank, world_size, device):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)

class WorkerExtension:
    """
    Methods used by the trainer:
    - perturb_self_weights(seed, sigma_or_scale, coeff=1.0, negate=False)
    - restore_self_weights(seed, SIGMA)
    - init_inter_engine_group(master_address, master_port, rank, world_size)
    - broadcast_all_weights(src_rank)
    - save_self_weights_to_disk(filepath)
    """

    def perturb_self_weights(self, seed, noise_scale, negate=False):
        scale = float(noise_scale)
        sign = -1.0 if negate else 1.0
        for _, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(sign * scale * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def restore_self_weights(self, seed, SIGMA):
        for _, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(-float(SIGMA) * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def init_inter_engine_group(self, master_address: str, master_port: int, rank: int, world_size: int):
        self.inter_pg = _stateless_init_process_group(
            master_address, master_port, rank, world_size, self.device
        )
        return True

    def broadcast_all_weights(self, src_rank: int):
        for _, p in self.model_runner.model.named_parameters():
            self.inter_pg.broadcast(p, src=int(src_rank), stream=torch.cuda.current_stream())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def save_self_weights_to_disk(self, filepath):
        state_dict_to_save = {}
        for name, p in self.model_runner.model.named_parameters():
            state_dict_to_save[name] = p.detach().cpu()
        torch.save(state_dict_to_save, filepath)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.1)
        return True

    def get_weights_to_cpu(self):
        """Return a CPU copy of all model parameters as a state dict."""
        return {name: p.detach().cpu() for name, p in self.model_runner.model.named_parameters()}

    def apply_delta_from_cpu(self, other_weights, migration_lr):
        """Apply θ += migration_lr * (other_weights - θ) for each parameter."""
        scale = float(migration_lr)
        for name, p in self.model_runner.model.named_parameters():
            if name in other_weights:
                other = other_weights[name].to(p.device, dtype=p.dtype)
                p.data.add_(scale * (other - p.data))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def load_weights_from_disk(self, filepath):
        """Load a state dict saved by save_self_weights_to_disk and overwrite current weights."""
        state_dict = torch.load(filepath, map_location="cpu")
        for name, p in self.model_runner.model.named_parameters():
            if name in state_dict:
                p.data.copy_(state_dict[name].to(p.device, dtype=p.dtype))
        del state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return True
