"""Select one task and save a complete native FastWAM evaluation."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import urllib.request
ROOT = Path(__file__).resolve().parents[2]
ALL_TASKS = {p.stem for p in (ROOT / 'configs/multi_task').glob('*.yaml') if p.stem.startswith(('t', 'eval_'))}
TASKS = {int(name[1:3]): name for name in ALL_TASKS if name.startswith('t') and name[1:3].isdigit()}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, help='Task number, e.g. 42, or full task identifier')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--policy-url', default=os.environ.get('HANDSON_POLICY_URL', 'http://127.0.0.1:8631'))
    args = parser.parse_args()
    task = TASKS.get(int(args.task)) if args.task.isdigit() else args.task
    if task not in ALL_TASKS: parser.error('Unknown task identifier')
    output = args.output or Path('outputs') / f"fastwam-{task}-{datetime.now():%Y%m%d_%H%M%S_%f}"
    if not output.is_absolute(): output = ROOT / output
    if output.exists(): parser.error('Output directory must not already exist')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(args.policy_url, timeout=30) as response: metadata = json.load(response)
    if metadata.get('model') != 'fastwam': parser.error('Endpoint is not FastWAM')
    if metadata.get('task', task) != task: parser.error('Server is bound to a different task')
    if metadata.get('checkpoint_sha256') != '9d8532d0ad8b62596ba070f7f663c9e58d67a059523b7fe4c21ea7028bedf46f':
        parser.error('Unexpected checkpoint identity')
    receipts = ROOT / 'local_runtime/policy_metadata'; receipts.mkdir(parents=True, exist_ok=True)
    receipt = receipts / f'{task}-{datetime.now():%Y%m%d_%H%M%S_%f}.json'
    with receipt.open('x') as stream: json.dump(metadata, stream, indent=2)
    print(f'Task: {task}\nOutput: {output}', flush=True)
    command = ['bash', str(ROOT / 'run_handson_sim.sh'), '-m', 'handson.evaluation', '--model', 'fastwam',
               '--task', task, '--policy-url', args.policy_url, '--policy-metadata', str(receipt),
               '--seed', str(args.seed), '--output', str(output)]
    raise SystemExit(subprocess.call(command, cwd=ROOT))
if __name__ == '__main__': main()
