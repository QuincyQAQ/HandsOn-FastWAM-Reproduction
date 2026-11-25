"""Run an isolated archived trial with dense original 30Hz capture and verify video receipts."""
from pathlib import Path
import datetime,json,subprocess,os,shutil,hashlib,traceback
ROOT=Path(__file__).resolve().parents[2]
BATCH=ROOT/'outputs'/('highfps-t45_stir_pot-trial000-'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
BATCH.mkdir(parents=True,exist_ok=False);OUT=BATCH/'episode';STATUS=BATCH/'STATUS.json'
def status(stage,**kw):
 d=dict(stage=stage,updated_at=datetime.datetime.now().isoformat(),output=str(OUT),**kw);STATUS.write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
try:
 meta=ROOT/'归档实验/remaining54-paper-trial000-20261109_190242/POLICY_METADATA.json';shutil.copy2(meta,BATCH/'POLICY_METADATA.json')
 source=BATCH/'source';source.mkdir();hashes={}
 for n in ['cli.py','native.py','rollout.py','schedule.py']:
  src=ROOT/'handson/handson/evaluation'/n;shutil.copy2(src,source/n);hashes[n]=hashlib.sha256(src.read_bytes()).hexdigest()
 (BATCH/'SOURCE_SHA256.json').write_text(json.dumps(hashes,indent=2))
 cmd=['bash',str(ROOT/'run_handson_sim.sh'),'-m','handson.evaluation','--model','fastwam','--task','t45_stir_pot','--paper-trial','0','--record-stride','1','--policy-url','http://127.0.0.1:8633','--policy-metadata',str(BATCH/'POLICY_METADATA.json'),'--output',str(OUT)]
 (BATCH/'COMMAND.json').write_text(json.dumps(cmd,indent=2));env=dict(os.environ)
 for k in list(env):
  if k.startswith('HANDSON_LIVE_'):env.pop(k)
 env['HANDSON_LIVE_VIEW']='0'
 status('running',purpose='Original dense capture, no frame interpolation; original policy horizon16 and archived trial0')
 with (BATCH/'launcher.log').open('w') as log:completed=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=28800)
 if completed.returncode:raise RuntimeError(f'Evaluator exited {completed.returncode}; inspect launcher.log and episode/runtime.log')
 result=json.loads((OUT/'policy_000/RESULT.json').read_text());steps=result['steps'];assert result['complete_evaluation_episode']
 assert result['frame_steps']==list(range(steps+1)),'Missing original per-step captures'
 checks={}
 for name in ['main','wrist_left','wrist_right']:
  video=OUT/'policy_000'/f'{name}.mp4'
  info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=avg_frame_rate,nb_frames,duration','-of','json',str(video)]))['streams'][0]
  num,den=map(float,info['avg_frame_rate'].split('/'));assert abs(num/den-30)<1e-5;assert int(info['nb_frames'])==steps+1
  subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
  checks[name]=info
 receipt=dict(steps=steps,success=result['success'],original_capture=True,interpolated=False,video_checks=checks,policy_render_stride=16,record_stride=1,output=str(OUT),scope='New live-policy repeat, not guaranteed identical trajectory to old success and not merged into old benchmark')
 (BATCH/'VIDEO_VERIFICATION.json').write_text(json.dumps(receipt,indent=2));status('complete',**{k:v for k,v in receipt.items() if k!='output'})
 # Replace presentation background only after a complete successful native trial.
 if result['success'] and not (ROOT/'local_runtime/KEEP_ORIGINAL_HERO.json').exists():
  site=ROOT/'网站版本/v2.0';target=site/'project-assets/intro-background-native30.mp4'
  subprocess.run(['ffmpeg','-y','-v','error','-i',str(OUT/'policy_000/main.mp4'),'-c','copy','-movflags','+faststart',str(target)],check=True)
  page=site/'index.html';s=page.read_text();s=s.replace('project-assets/intro-background-smooth.mp4','project-assets/intro-background-native30.mp4').replace('双臂操作 · 仿真录像（插帧展示）','双臂操作 · 原始30fps录像');page.write_text(s)
  (site/'project-assets/intro-native30-source.json').write_text(json.dumps(receipt,indent=2))
 with (ROOT/'RBMproject.MD').open('a') as f:f.write('\n\n### 原始高采样录像完成\n\n输出：`'+str(BATCH.relative_to(ROOT))+'`。每控制步真实采集一次、30fps，三路均通过逐帧/解码验收；原生判定成功：'+str(result['success'])+'。保留原权重、缓存、trial0、控制与物理条件；新增独立录像stride1，原策略图像边界stride16。不是插帧，也不保证重复出完全相同轨迹。详细见VIDEO_VERIFICATION.json。\n')
except Exception as e:
 status('failed',error=str(e));(BATCH/'ERROR.txt').write_text(traceback.format_exc());raise
