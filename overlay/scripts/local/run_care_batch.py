"""Prepare an isolated care-task service, then run the requested 12 tasks serially."""
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import urllib.request
ROOT = Path(__file__).resolve().parents[2]
TASK_IDS = [42,46,25,28,5,19,15,21,26,17,10,3]
TASKS = {int(p.name[1:3]):p.stem for p in (ROOT/'configs/multi_task').glob('t[0-9][0-9]_*.yaml')}
RUN = ROOT/'outputs'/f'care-batch-{datetime.now():%Y%m%d_%H%M%S_%f}'
RUN.mkdir(parents=True)
print(f'Batch output: {RUN}', flush=True)
spec=importlib.util.spec_from_file_location('tunnel',ROOT/'local_runtime/fastwam_tunnel.py')
tunnel=importlib.util.module_from_spec(spec); spec.loader.exec_module(tunnel)
client=tunnel.connect()
remote='/data1/qli736/robot/HandsOn-main'
stamp=RUN.name
cache=f'/data1/qli736/robot/weights/handson/fastwam/{stamp}.safetensors'
metadata=f'{remote}/outputs/{stamp}-metadata.json'
remote_log=f'{remote}/outputs/{stamp}-server.log'
prepare=' '.join('--task '+shlex.quote(TASKS[n]) for n in TASK_IDS)
script=f'''#!/bin/bash
set -euo pipefail
cd {remote}
bash run_handson_npu.sh -m handson.policies.encode_text --model fastwam --device npu:9 --encoder /data1/qli736/robot/weights/handson/wan-text/models_t5_umt5-xxl-enc-bf16.pth --tokenizer /data1/qli736/robot/weights/handson/wan-text/google/umt5-xxl {prepare} --output {cache}
HANDSON_FASTWAM_DEVICE=npu:9 bash run_handson_npu.sh scripts/serve_paper_policy.py --model fastwam --checkpoint /data1/qli736/robot/data/weights/handson/fastwam/step-100000/mp_rank_00_model_states.pt --checkpoint-sha256 9d8532d0ad8b62596ba070f7f663c9e58d67a059523b7fe4c21ea7028bedf46f --text-cache {cache} --bind 127.0.0.1 --port 8631 --metadata-output {metadata}
'''
# Refuse to collide with another service or an occupied NPU.
check="ss -ltn | grep -q ':8631 ' && echo PORT_BUSY || true; npu-smi info -i 4"
_,stdout,stderr=client.exec_command(check); diagnostic=stdout.read().decode(); print(diagnostic,flush=True)
if 'PORT_BUSY' in diagnostic: raise RuntimeError('Port 8631 already in use; refusing to replace service')
remote_script=f'{remote}/outputs/{stamp}.sh'
with client.open_sftp() as sftp:
 with sftp.open(remote_script,'w') as f: f.write(script)
command=f'tmux new-session -d -s {shlex.quote(stamp)} '+shlex.quote(f'bash {remote_script} > {remote_log} 2>&1')
_,stdout,stderr=client.exec_command(command)
if stdout.channel.recv_exit_status(): raise RuntimeError(stderr.read().decode())
(RUN/'BATCH.json').write_text(json.dumps(dict(tasks=[TASKS[n] for n in TASK_IDS],seed=0,remote_log=remote_log,remote_session=stamp,cache=cache),indent=2))
# A dedicated forwarding process leaves the original cup service/tunnel untouched.
forward=subprocess.Popen([sys.executable,str(ROOT/'local_runtime/fastwam_tunnel.py'),'--port','8631'])
try:
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 deadline=time.monotonic()+7200
 while True:
  if forward.poll() is not None: raise RuntimeError('SSH forwarding exited')
  try:
   with opener.open('http://127.0.0.1:8631',timeout=5) as response: info=json.load(response)
   if info.get('model')=='fastwam': break
  except Exception: pass
  if time.monotonic()>deadline: raise TimeoutError('Text cache/server startup timeout; see remote log')
  print('Waiting for authentic text cache and NPU model service...',flush=True); time.sleep(30)
 results=[]
 for number in TASK_IDS:
  task=TASKS[number]; output=RUN/f'fastwam-{task}-seed0'
  print(f'START {task}',flush=True)
  with (RUN/f'{task}.log').open('w') as log:
   code=subprocess.call(['bash',str(ROOT/'run_fastwam_task.sh'),'--task',task,'--seed','0','--output',str(output)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
  result=json.loads((output/'RESULT.json').read_text()) if (output/'RESULT.json').exists() else {}
  results.append(dict(task=task,exit_code=code,success=result.get('success'),steps=result.get('steps'),output=str(output)))
  (RUN/'SUMMARY.json').write_text(json.dumps(results,indent=2))
  print(json.dumps(results[-1]),flush=True)
 print('BATCH COMPLETE',flush=True)
finally:
 forward.terminate(); forward.wait(timeout=30); client.close()
