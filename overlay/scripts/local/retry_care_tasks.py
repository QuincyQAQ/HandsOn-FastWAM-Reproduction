"""Retry the previous unsuccessful/errored care tasks with seed1, preserving originals."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
ROOT=Path(__file__).resolve().parents[2]
PRIOR=ROOT/'outputs/care-batch-20261106_152040_147176'
TASKS=[r for r in json.loads((PRIOR/'SUMMARY.json').read_text()) if r['success'] is not True]
RUN=ROOT/'outputs'/f'care-retry-{datetime.now():%Y%m%d_%H%M%S_%f}'
RUN.mkdir(parents=True)
(RUN/'BATCH.json').write_text(json.dumps(dict(source=str(PRIOR),seed=1,tasks=[r['task'] for r in TASKS],note='One new-seed attempt each; original native success checks retained.'),indent=2))
print(f'Batch output: {RUN}',flush=True)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
forward=None
try:
 try:
  with opener.open('http://127.0.0.1:8631',timeout=5) as response:metadata=json.load(response)
 except Exception:
  forward=subprocess.Popen([sys.executable,str(ROOT/'local_runtime/fastwam_tunnel.py'),'--port','8631'])
  for i in range(30):
   if forward.poll() is not None:raise RuntimeError('Forwarder exited')
   try:
    with opener.open('http://127.0.0.1:8631',timeout=5) as response:metadata=json.load(response)
    break
   except Exception:
    if i==29:raise
    time.sleep(2)
 if metadata.get('model')!='fastwam' or metadata.get('task'):raise RuntimeError('Requires the unbound multi-task FastWAM service')
 results=[]
 for prior in TASKS:
  task=prior['task'];output=RUN/f'fastwam-{task}-seed1'
  print(f'START {task} seed1',flush=True)
  with (RUN/f'{task}.log').open('x') as log:
   code=subprocess.call(['bash',str(ROOT/'run_fastwam_task.sh'),'--task',task,'--seed','1','--output',str(output)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  result=json.loads((output/'RESULT.json').read_text()) if (output/'RESULT.json').exists() else {}
  error=json.loads((output/'ERROR.json').read_text()).get('error') if (output/'ERROR.json').exists() else None
  row=dict(task=task,seed=1,exit_code=code,success=result.get('success'),steps=result.get('steps'),error=error,output=str(output))
  results.append(row)
  temporary=RUN/'SUMMARY.json.tmp';temporary.write_text(json.dumps(results,indent=2));temporary.replace(RUN/'SUMMARY.json')
  print(json.dumps(row),flush=True)
  # New successful examples are added alongside the original collection.
  if result.get('success') is True:
   import shutil
   collection=ROOT/'main视频集合';collection.mkdir(exist_ok=True)
   old=next(collection.glob(task[1:3]+'_*.mp4'),None)
   name=f'{old.stem if old else task}_seed1_成功.mp4'
   target=collection/name
   if not target.exists():shutil.copy2(output/'policy_000/main.mp4',target)
 print('BATCH COMPLETE',flush=True)
finally:
 if forward:
  forward.terminate();forward.wait(timeout=30)
