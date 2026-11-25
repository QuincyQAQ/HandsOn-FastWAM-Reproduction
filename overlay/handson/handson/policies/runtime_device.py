"""Local optional Ascend support; CUDA remains the release default."""
import os


def fastwam_device():
    import torch
    device = torch.device(os.environ.get("HANDSON_FASTWAM_DEVICE", "cuda:0"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type == "npu":
        import torch_npu
        torch.npu.set_device(device)
        from .common import vendor
        vendor("fastwam")
        from fastwam.models.wan22 import wan_video_dit as dit
        if not getattr(dit, "_handson_npu", False):
            frequency = dit.precompute_freqs_cis
            sinusoidal = dit.sinusoidal_embedding_1d
            rope = dit.rope_apply

            def npu_frequency(*args, **kwargs):
                # Native CPU double construction, then NPU-supported storage.
                return frequency(*args, **kwargs).to(torch.complex64)

            def npu_sinusoidal(dim, position):
                if position.device.type != "npu":
                    return sinusoidal(dim, position)
                return sinusoidal(dim, position.cpu()).to(position.device)

            def npu_rope(x, freqs, num_heads):
                if x.device.type != "npu":
                    return rope(x, freqs, num_heads)
                batch, seq, _ = x.shape
                pairs = x.float().reshape(batch, seq, num_heads, -1, 2)
                # CPU view avoids unsupported complex NPU kernels.
                phase = torch.view_as_real(freqs.cpu()).to(x.device)
                a, b = pairs[..., 0], pairs[..., 1]
                c, d = phase[..., 0], phase[..., 1]
                return torch.stack((a*c - b*d, a*d + b*c), dim=-1).flatten(2).to(x.dtype)

            dit.precompute_freqs_cis = npu_frequency
            dit.sinusoidal_embedding_1d = npu_sinusoidal
            dit.rope_apply = npu_rope
            from fastwam.models.wan22 import action_dit, mot
            action_dit.precompute_freqs_cis = npu_frequency
            action_dit.sinusoidal_embedding_1d = npu_sinusoidal
            action_dit.rope_apply = npu_rope
            mot.rope_apply = npu_rope
            dit._handson_npu = True
    elif device.type != "cpu":
        raise ValueError(f"Unsupported FastWAM device: {device}")
    return device
