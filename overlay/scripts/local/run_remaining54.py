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
TASK_IDS = json.loads((ROOT/'local_runtime/remaining54-tasks.json').read_text())
TASKS = {task: task for task in TASK_IDS}
RUN = ROOT/'outputs'/f'all54-batch-{datetime.now():%Y%m%d_%H%M%S_%f}'
RUN.mkdir(parents=True)
print(f'Batch output: {RUN}', flush=True)
(RUN/'QUEUE.json').write_text(json.dumps(dict(tasks=TASK_IDS,seed=0,wait_for='handson-care-retry-20261108.service',status='queued'),indent=2))
while subprocess.call(['systemctl','--user','is-active','--quiet','handson-care-retry-20261108.service']) == 0:
 print('QUEUED: waiting for the current retry batch to finish',flush=True)
 time.sleep(30)
print('Previous retry service ended; preparing remaining54 tasks',flush=True)
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
bash run_handson_npu.sh -m handson.policies.encode_text --model fastwam --allow-native-truncation --device npu:10 --encoder /data1/qli736/robot/weights/handson/wan-text/models_t5_umt5-xxl-enc-bf16.pth --tokenizer /data1/qli736/robot/weights/handson/wan-text/google/umt5-xxl {prepare} --output {cache}
HANDSON_FASTWAM_DEVICE=npu:10 bash run_handson_npu.sh scripts/serve_paper_policy.py --model fastwam --checkpoint /data1/qli736/robot/data/weights/handson/fastwam/step-100000/mp_rank_00_model_states.pt --checkpoint-sha256 9d8532d0ad8b62596ba070f7f663c9e58d67a059523b7fe4c21ea7028bedf46f --text-cache {cache} --bind 127.0.0.1 --port 8632 --metadata-output {metadata}
'''
# Refuse to collide with another service or an occupied NPU.
check="ss -ltn | grep -q ':8632 ' && echo PORT_BUSY || true; npu-smi info -i 5 -c 0"
_,stdout,stderr=client.exec_command(check); diagnostic=stdout.read().decode(); print(diagnostic,flush=True)
if 'PORT_BUSY' in diagnostic: raise RuntimeError('Port 8632 already in use; refusing to replace service')
remote_script=f'{remote}/outputs/{stamp}.sh'
with client.open_sftp() as sftp:
 with sftp.open(remote_script,'w') as f: f.write(script)
command=f'tmux new-session -d -s {shlex.quote(stamp)} '+shlex.quote(f'bash {remote_script} > {remote_log} 2>&1')
_,stdout,stderr=client.exec_command(command)
if stdout.channel.recv_exit_status(): raise RuntimeError(stderr.read().decode())
(RUN/'BATCH.json').write_text(json.dumps(dict(tasks=[TASKS[n] for n in TASK_IDS],seed=0,exploratory_native_text_truncation=True,remote_log=remote_log,remote_session=stamp,cache=cache),indent=2))
# A dedicated forwarding process leaves the original cup service/tunnel untouched.
forward=subprocess.Popen([sys.executable,str(ROOT/'local_runtime/fastwam_tunnel.py'),'--port','8632'])
try:
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 deadline=time.monotonic()+1800
 while True:
  if forward.poll() is not None: raise RuntimeError('SSH forwarding exited')
  try:
   with opener.open('http://127.0.0.1:8632',timeout=5) as response: info=json.load(response)
   if info.get('model')=='fastwam': break
  except Exception: pass
  _, state_out, _ = client.exec_command('tmux has-session -t ' + shlex.quote(stamp))
  if state_out.channel.recv_exit_status() != 0:
   _, failed_log, _ = client.exec_command('tail -n 40 ' + shlex.quote(remote_log))
   raise RuntimeError('Remote preparation exited: ' + failed_log.read().decode())
  if time.monotonic()>deadline: raise TimeoutError('Text cache/server startup timeout; see remote log')
  print('Waiting for authentic text cache and NPU model service...',flush=True); time.sleep(30)
 results=[]
 for number in TASK_IDS:
  task=TASKS[number]; output=RUN/f'fastwam-{task}-seed0'
  print(f'START {task}',flush=True)
  with (RUN/f'{task}.log').open('w') as log:
   code=subprocess.call(['bash',str(ROOT/'run_fastwam_task.sh'),'--task',task,'--seed','0','--policy-url','http://127.0.0.1:8632','--output',str(output)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
  result=json.loads((output/'RESULT.json').read_text()) if (output/'RESULT.json').exists() else {}
  video_checks = {}
  for view in ('main','wrist_left','wrist_right'):
   video = output/'policy_000'/f'{view}.mp4'
   check = subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','json',str(video)],capture_output=True,text=True)
   video_checks[view] = dict(parse_ok=check.returncode==0,details=json.loads(check.stdout) if check.returncode==0 else check.stderr)
  complete = bool(code==0 and result.get('complete_evaluation_episode') and all(v['parse_ok'] for v in video_checks.values()))
  results.append(dict(exploratory_native_text_truncation=True,task=task,exit_code=code,success=result.get('success'),steps=result.get('steps'),complete_video_episode=complete,video_checks=video_checks,output=str(output)))
  (RUN/'SUMMARY.json').write_text(json.dumps(results,indent=2))
  print(json.dumps(results[-1]),flush=True)
 print('BATCH COMPLETE',flush=True)
finally:
 forward.terminate(); forward.wait(timeout=30); client.close()
