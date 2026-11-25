"""Full archived cup trial0, strict language cache and live control guard."""
from pathlib import Path
import importlib.util,json,subprocess,sys,time,urllib.request,shlex
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'归档实验/runs'/f'fastwam-t25_place_cup_on_plate-trial000-{datetime.now():%Y%m%d_%H%M%S}'
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
 receipt=CONTROL/'PHYSX_POLICY_METADATA.json';receipt.write_text(json.dumps(info,indent=2))
 command=['bash',str(ROOT/'run_handson_sim.sh'),'-m','handson.evaluation','--model','fastwam','--task','t25_place_cup_on_plate','--paper-trial','0','--policy-url','http://127.0.0.1:8633','--policy-metadata',str(receipt),'--output',str(RUN)]
 code=subprocess.call(command,cwd=ROOT)
 result=json.loads((RUN/'RESULT.json').read_text()) if (RUN/'RESULT.json').exists() else {}
 report=dict(output=str(RUN),exit_code=code,success=result.get('success'),steps=result.get('steps'),complete_evaluation_episode=result.get('complete_evaluation_episode'),control_backend=json.loads((RUN/'ARCHIVED_CONTROL_BACKEND.json').read_text()) if (RUN/'ARCHIVED_CONTROL_BACKEND.json').exists() else None)
 if (RUN/'FINAL_STATE.json').exists():
  final=json.loads((RUN/'FINAL_STATE.json').read_text());report['final_haorui_enabled']=final.get('haorui_enabled');report['final_arm_ik_backend']=final.get('arm_ik_backend')
 report['video_checks']={}
 for view in ['main','wrist_left','wrist_right']:
  p=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','json',str(RUN/'policy_000'/f'{view}.mp4')],capture_output=True,text=True)
  report['video_checks'][view]=dict(exit_code=p.returncode,info=json.loads(p.stdout) if p.returncode==0 else p.stderr)
 (CONTROL/'PHYSX_VALIDATION_RESULT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
finally:
 forward.terminate();forward.wait(timeout=30);client.close()
