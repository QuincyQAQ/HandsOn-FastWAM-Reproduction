"""Local prerequisite check; does not certify a learned-policy rollout."""
import json
from pathlib import Path
import torch
import yaml
from handson.policies.runtime_device import fastwam_device
from handson.policies.common import vendor

device = fastwam_device()
vendor('fastwam')
from fastwam.models.wan22 import wan_video_dit as dit
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_vae import WanVideoVAE38
from handson.policies.common import asset
from fastwam.models.wan22.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer

# Compare the device-specific rotary operation with native CPU double math.
torch.manual_seed(0)
x = torch.randn(1, 16, 128)
positions = torch.arange(32).double()
angles = torch.outer(torch.arange(16).double(), 10000 ** (-positions / 32))
freqs = torch.polar(torch.ones_like(angles), angles).view(16, 1, 32)
reference = torch.view_as_real(torch.view_as_complex(x.double().reshape(1,16,2,32,2)) * freqs).flatten(2).float()
actual = dit.rope_apply(x.to(device), freqs.to(torch.complex64).to(device), 2).cpu()
torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-5)
embedding = dit.sinusoidal_embedding_1d(256, torch.tensor([1.0], device=device))
assert torch.isfinite(embedding).all()
q = torch.randn(1, 2, 16, 64, device=device)
attention = torch.nn.functional.scaled_dot_product_attention(q, q, q)
assert torch.isfinite(attention).all()
checkpoint = Path('/data1/qli736/robot/data/weights/handson/fastwam/step-100000/mp_rank_00_model_states.pt')
ck = torch.load(str(checkpoint), map_location='cpu', mmap=True, weights_only=False)
assert ck['optimizer_step'] == ck['global_steps'] == 100000
assert ck['prompt_patch'] == '03'
assert ck['data_fingerprint'] == '17692771a13cd04a94568df00d2e97edc2bc1a90a2d54c6f1569e2147c914589'
assert ck['excluded_tasks'] == ['h01_chinese_chess']
config = yaml.safe_load(asset('fastwam', 'model.yaml').read_text())
# Meta tensors validate every architecture key and shape without allocating a
# second full CPU/GPU copy. No inference is performed with this meta model.
with torch.device('meta'):
    video = WanVideoDiT(**config['video_dit_config']).to(dtype=torch.bfloat16)
    action = ActionDiT(**config['action_dit_config']).to(dtype=torch.bfloat16)
    vae = WanVideoVAE38().to(dtype=torch.bfloat16)
# Scheduler initialization uses scalar CPU reductions, so leave the meta
# construction context before assembling the container and its tiny encoders.
model = FastWAM(video, action, MoT({'video':video,'action':action}), vae,
    text_dim=4096, proprio_dim=52, device='meta', torch_dtype=torch.bfloat16,
    video_train_shift=config['video_scheduler']['train_shift'],
    video_infer_shift=config['video_scheduler']['infer_shift'],
    video_num_train_timesteps=config['video_scheduler']['num_train_timesteps'],
    action_train_shift=config['action_scheduler']['train_shift'],
    action_infer_shift=config['action_scheduler']['infer_shift'],
    action_num_train_timesteps=config['action_scheduler']['num_train_timesteps'])
loaded = model.load_state_dict(ck['module'], strict=True, assign=True)
assert not loaded.missing_keys and not loaded.unexpected_keys
print(json.dumps(dict(device=str(device), torch=torch.__version__,
    checkpoint_step=ck['global_steps'], module_keys=len(ck['module']),
    architecture_strict_load=True,
    rope_max_error=float((actual-reference).abs().max()),
    scope='imports, NPU rotary/attention operators, checkpoint metadata and strict architecture load; no inference'), indent=2))
