"""Audit completed archived PhysX control evidence without changing outcomes."""
import json,math,sys
from pathlib import Path
folder=Path(sys.argv[1])
result=json.loads((folder/'RESULT.json').read_text())
backend=json.loads((folder/'ARCHIVED_CONTROL_BACKEND.json').read_text())
final=json.loads((folder/'FINAL_STATE.json').read_text())
first=json.loads((folder/'policy_000/FIRST_COMMAND_RECEIVED.json').read_text())
actions=[json.loads(s) for s in (folder/'policy_000/steps.jsonl').read_text().splitlines()]
log=(folder/'runtime.log').read_text(errors='replace')
checks=dict(complete_evaluation_episode=result.get('complete_evaluation_episode') is True,
 actual_haorui_initialized=backend.get('actual')=='haorui' and backend.get('haorui_enabled') is True,
 final_haorui_enabled=final.get('haorui_enabled') is True,
 final_backend_haorui=final.get('arm_ik_backend')=='haorui',
 exact_first_command=all(v['position_max_abs']==0 and v['hand_max_abs']==0 and v['rotation_geodesic_rad']==0 for v in first.values()),
 action_count_matches=len(actions)==result['steps'],
 actions_finite=all(math.isfinite(v) for a in actions for v in a['action']),
 no_dls_fallback='haorui_fallback_dls' not in log and 'Haorui ROKAE IK unavailable' not in log,
 recorded_archived_inputs=(folder/'PAPER_TRIAL.json').is_file(),
 no_runtime_error=not (folder/'ERROR.json').exists())
report=dict(output=str(folder),checks=checks,control_evidence_pass=all(checks.values()),native_task_success=result['success'],steps=result['steps'],hold_log_lines=sum('haorui_hold' in l for l in log.splitlines()),scope='One complete archived cup trial; does not prove all-task control or CUDA/NPU parity')
p=Path(__file__).with_name('PHYSX_CONTROL_AUDIT.json');p.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
