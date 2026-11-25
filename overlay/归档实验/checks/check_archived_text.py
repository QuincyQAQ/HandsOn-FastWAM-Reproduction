"""Read-only: validate Core50 trial0 prompts with the native tokenizer and existing caches."""
import hashlib,json
from pathlib import Path
import torch
from handson.policies.common import vendor
from handson.evaluation.paper_trials import manifest,load
vendor('fastwam')
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from safetensors import safe_open
root=Path('/data1/qli736/robot/weights/handson')
tokenizer=HuggingfaceTokenizer(str(root/'wan-text/google/umt5-xxl'),seq_len=128,clean='whitespace')
rows=[]
for task in manifest()['tasks']:
 row=load('fastwam',task,0);prompt=row['prompt'];wrapped="A video recorded from a robot's point of view executing the following instruction: "+prompt
 count=len(tokenizer.tokenizer(tokenizer._clean(wrapped),add_special_tokens=True)['input_ids'])
 rows.append(dict(task=task,trial=0,prompt=prompt,key=hashlib.sha256(prompt.encode()).hexdigest(),tokens=count,within_limit=count<=128,cache_matches=[]))
for cache in sorted((root/'fastwam').glob('*.safetensors')):
 receipt=cache.with_suffix(cache.suffix+'.json');info=json.loads(receipt.read_text()) if receipt.exists() else {}
 with safe_open(str(cache),framework='pt',device='cpu') as f:
  keys=set(f.keys())
  for row in rows:
   if row['key'] not in keys:continue
   tensor=f.get_tensor(row['key']);valid=tensor.ndim==2 and list(tensor.shape)==[row['tokens'],4096] and bool(torch.isfinite(tensor).all())
   row['cache_matches'].append(dict(path=str(cache),shape=list(tensor.shape),valid=valid,encoder_sha256=info.get('encoder_sha256'),native_truncation_allowed=info.get('native_truncation_allowed',False)))
report=dict(scope='FastWAM Core50 paper-trial0 only',limit=128,total=len(rows),over_limit=sum(not r['within_limit'] for r in rows),covered_exact=sum(any(m['valid'] for m in r['cache_matches']) for r in rows),rows=rows)
p=Path('/data1/qli736/robot/HandsOn-main/outputs/archive-text-check-trial0.json');p.write_text(json.dumps(report,indent=2));print(json.dumps(report))
# Assemble only the exact, untruncated archived prompts; never alter existing caches.
from safetensors.torch import save_file
strict=root/'fastwam/archive-core50-trial000-strict.safetensors'
if report['over_limit'] or report['covered_exact']!=50:raise RuntimeError('Strict archived cache preflight failed')
values={}
for row in rows:
 match=next(m for m in row['cache_matches'] if m['valid'] and m['encoder_sha256']=='7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d')
 with safe_open(match['path'],framework='pt',device='cpu') as f:values[row['key']]=f.get_tensor(row['key']).clone().contiguous()
if strict.exists():
 with safe_open(str(strict),framework='pt',device='cpu') as f:
  if set(f.keys())!=set(values) or not all(torch.equal(f.get_tensor(k),v) for k,v in values.items()):raise RuntimeError('Existing strict cache differs')
else:save_file(values,str(strict))
receipt=dict(scope='FastWAM Core50 paper-trial0 exact prompts',prompts=len(values),native_truncation_allowed=False,token_limit=128,encoder_sha256='7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d',cache_sha256=hashlib.sha256(strict.read_bytes()).hexdigest(),cache=str(strict),source_report=str(p))
strict.with_suffix('.safetensors.json').write_text(json.dumps(receipt,indent=2));print('STRICT_CACHE_RECEIPT '+json.dumps(receipt))
