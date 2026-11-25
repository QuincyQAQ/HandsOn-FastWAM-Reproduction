"""Deterministic cross-device primitive probes; does not establish full-policy parity."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from handson.policies.common import vendor
p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--output',required=True);a=p.parse_args()
vendor('fastwam')
if a.device.startswith('npu'):
 import os
 os.environ['HANDSON_FASTWAM_DEVICE']=a.device
 from handson.policies.runtime_device import fastwam_device
 device=fastwam_device()
else:device=torch.device(a.device);torch.cuda.set_device(device)
from fastwam.models.wan22 import wan_video_dit as dit
rng=np.random.default_rng(20261109);outputs={}
for seed in range(3):
 x=rng.standard_normal((1,64,3072),dtype=np.float32)
 for dtype in [torch.float32,torch.bfloat16]:
  suffix=f'{seed}_{str(dtype).split(".")[-1]}'
  freq=dit.precompute_freqs_cis(128,end=64).reshape(64,1,64).to(device)
  y=dit.rope_apply(torch.from_numpy(x).to(device=device,dtype=dtype),freq,24)
  outputs['rope_'+suffix]=y.float().cpu().numpy()
  qkv=[torch.from_numpy(rng.standard_normal((1,4,32,128),dtype=np.float32)).to(device=device,dtype=dtype) for _ in range(3)]
  y=torch.nn.functional.scaled_dot_product_attention(*qkv,dropout_p=0.0)
  outputs['sdpa_'+suffix]=y.float().cpu().numpy()
 pos=torch.tensor([0.,1.,37.,999.],device=device)
 outputs[f'sinusoidal_{seed}']=dit.sinusoidal_embedding_1d(256,pos).float().cpu().numpy()
np.savez(a.output,**outputs)
Path(a.output+'.json').write_text(json.dumps(dict(device=a.device,torch=torch.__version__,probes=list(outputs),scope='RoPE, SDPA, sinusoidal fixed generated inputs; not model inference'),indent=2))
print('SAVED '+a.output,flush=True)
