"""Full archived cup trial0, strict language cache and live control guard."""
from pathlib import Path
import importlib.util,json,subprocess,sys,time,urllib.request,shlex
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'归档实验'/f'remaining54-paper-trial000-{datetime.now():%Y%m%d_%H%M%S}'
RUN.mkdir(parents=True)
remaining=json.loads((ROOT/'local_runtime/remaining54-tasks.json').read_text())
archived=set(json.loads((ROOT/'handson/handson/evaluation/paper_trials.json').read_text())['tasks'])
selected=[task for task in remaining if task in archived]
unsupported=[task for task in remaining if task not in archived]
(RUN/'PLAN.json').write_text(json.dumps(dict(requested=remaining,archived_trial0_tasks=selected,no_archived_trial=unsupported,trial=0,no_seed_substitution=True),indent=2))
print(f'Archived tasks: {len(selected)}; no archived inputs: {len(unsupported)}',flush=True)
CONTROL=ROOT/'归档实验/checks'
print('OUTPUT '+str(RUN),flush=True)
spec=importlib.util.spec_from_file_location('tunnel',ROOT/'local_runtime/fastwam_tunnel.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);client=t.connect()
remote='/data1/qli736/robot/HandsOn-main';metadata=f'{remote}/outputs/archive-physx-{datetime.now():%Y%m%d_%H%M%S}-metadata.json'
_,out,_=client.exec_command("ss -ltn | grep ':8633 ' || true");existing=out.read().decode()
if not existing:
 command=f'cd {remote} && HANDSON_FASTWAM_DEVICE=npu:11 bash run_handson_npu.sh scripts/serve_paper_policy.py --model fastwam --checkpoint /data1/qli736/robot/data/weights/handson/fastwam/step-100000/mp_rank_00_model_states.pt --checkpoint-sha256 9d8532d0ad8b62596ba070f7f663c9e58d67a059523b7fe4c21ea7028bedf46f --text-cache /data1/qli736/robot/weights/handson/fastwam/archive-core50-trial000-strict.safetensors --bind 127.0.0.1 --port 8633 --metadata-output {metadata} > {remote}/outputs/archive-physx-server.log 2>&1'
 _,out,err=client.exec_command('tmux new-session -d -s handson-archive-physx '+shlex.quote(command))
 if out.channel.recv_exit_status():raise RuntimeError(err.read().decode())
forward=subprocess.Popen([sys.executable,str(ROOT/'local_runtime/fastwam_tunnel.py'),'--port','8633'])
try:
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));deadline=time.monotonic()+900
 while True:
  try:
   with opener.open('http://127.0.0.1:8633',timeout=5) as response:info=json.load(response)
   break
  except Exception:
   if time.monotonic()>deadline:raise TimeoutError('Archived NPU server failed to become ready')
   print('Waiting for strict archived policy service',flush=True);time.sleep(15)
 expected=json.loads((CONTROL/'STRICT_CACHE_RECEIPT.json').read_text())['cache_sha256']
 assert info['text_cache_sha256']==expected and info['model']=='fastwam'
 receipt=RUN/'POLICY_METADATA.json';receipt.write_text(json.dumps(info,indent=2))
 results=[]
 for task in selected:
  output=RUN/f'fastwam-{task}-trial000'
  print(f'START {task} paper-trial0 -> {output}',flush=True)
  command=['bash',str(ROOT/'run_handson_sim.sh'),'-m','handson.evaluation','--model','fastwam','--task',task,'--paper-trial','0','--policy-url','http://127.0.0.1:8633','--policy-metadata',str(receipt),'--output',str(output)]
  with (RUN/f'{task}.log').open('x') as log:
   code=subprocess.call(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  result=json.loads((output/'RESULT.json').read_text()) if (output/'RESULT.json').exists() else {}
  error=json.loads((output/'ERROR.json').read_text()).get('error') if (output/'ERROR.json').exists() else None
  videos={}
  for view in ['main','wrist_left','wrist_right']:
   p=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','json',str(output/'policy_000'/f'{view}.mp4')],capture_output=True,text=True)
   count=None
   if p.returncode==0:
    streams=json.loads(p.stdout).get('streams',[])
    if streams and str(streams[0].get('nb_read_frames','')).isdigit():count=int(streams[0]['nb_read_frames'])
   videos[view]=dict(parse_ok=p.returncode==0,frames=count,matches_expected=count==len(result.get('frame_steps',[])) and bool(count))
  row=dict(task=task,paper_trial=0,exit_code=code,success=result.get('success'),steps=result.get('steps'),error=error,complete_video_episode=bool(code==0 and result.get('complete_evaluation_episode') and all(v['matches_expected'] for v in videos.values())),videos=videos,output=str(output))
  results.append(row)
  temporary=RUN/'SUMMARY.json.tmp';temporary.write_text(json.dumps(results,indent=2));temporary.replace(RUN/'SUMMARY.json')
  print(json.dumps(row),flush=True)
 print(f'ARCHIVED BATCH COMPLETE: {len(selected)} attempted; {len(unsupported)} have no archived Core trial and were not silently run with seed0',flush=True)
finally:
 forward.terminate();forward.wait(timeout=30);client.close()
