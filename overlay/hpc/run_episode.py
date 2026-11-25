"""Run a HandsOn episode against the allocated CUDA policy and verify three videos."""
import argparse
import json
import os
import shutil
from pathlib import Path
import subprocess
import urllib.request

CHECKPOINT = '9d8532d0ad8b62596ba070f7f663c9e58d67a059523b7fe4c21ea7028bedf46f'
ROOT = Path(__file__).resolve().parents[1]


def check_endpoint(url, receipt):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=20) as response:
        live = json.load(response)
    if (live != receipt or live.get('model') != 'fastwam'
            or live.get('checkpoint_sha256') != CHECKPOINT
            or live.get('load_receipt', {}).get('device') != 'cuda:0'
            or not live.get('text_cache_sha256')):
        raise ValueError('Endpoint differs from this job receipt or expected CUDA policy')


def verify_episode(output, receipt, task):
    result = json.loads((output / 'RESULT.json').read_text())
    if ((output / 'ERROR.json').exists() or not result['complete_evaluation_episode']
            or result['policy_metadata'] != receipt or result['task'] != task
            or not isinstance(result['success'], bool) or result['steps'] < 1):
        raise ValueError('Episode or policy identity could not be verified')
    for view in ('main', 'wrist_left', 'wrist_right'):
        path = output / 'policy_000' / (view + '.mp4')
        video = result['videos'][view]
        info = subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
            '-show_entries', 'stream=width,height,nb_read_frames', '-of', 'json', str(path)])
        stream, = json.loads(info)['streams']
        if (video['exit_code'] != 0 or video['frames'] < 1
                or video['frames'] != len(result['frame_steps'])
                or video['frames'] != int(stream['nb_read_frames'])
                or video['shape'] != [stream['height'], stream['width'], 3]):
            raise ValueError(f'Video differs from the episode record: {path}')
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(path),
                        '-map', '0:v:0', '-f', 'null', '-'], check=True)
    print(f"Verified three videos: {output} | task success={result['success']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--check-endpoint', action='store_true')
    parser.add_argument('--task')
    parser.add_argument('--output', type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--seed', type=int)
    selection.add_argument('--paper-trial', type=int)
    args = parser.parse_args()
    if not args.check_endpoint and not (args.task and args.output):
        parser.error('episode execution requires --task and --output')
    metadata = args.metadata.resolve()
    receipt = json.loads(metadata.read_text())
    check_endpoint(args.url, receipt)
    if args.check_endpoint:
        print('CUDA policy identity verified')
        return
    output = args.output.resolve()
    if output.exists():
        parser.error(f'choose a new episode directory: {output}')
    python = os.environ.get('HANDSON_SIM_PYTHON')
    if not python:
        parser.error('set HANDSON_SIM_PYTHON to the simulation interpreter or wrapper')
    for tool in ('ffmpeg', 'ffprobe'):
        if shutil.which(tool) is None:
            parser.error(f'install {tool} on the host that verifies the videos')
    env = os.environ.copy()
    paths = [str(ROOT / 'handson'), str(ROOT / 'third_party/IsaacLab-HandsOn/source/isaaclab')]
    if env.get('PYTHONPATH'):
        paths.append(env['PYTHONPATH'])
    env.update(PYTHONPATH=os.pathsep.join(paths), HANDSON_LIVE_VIEW='0',
               OMNI_KIT_ACCEPT_EULA='YES', PYTHONNOUSERSITE='1')
    trial = (['--paper-trial', str(args.paper_trial)] if args.paper_trial is not None
             else ['--seed', str(args.seed if args.seed is not None else 0)])
    subprocess.run([python, '-m', 'handson.evaluation', '--model', 'fastwam',
                    '--task', args.task, '--policy-url', args.url, '--policy-metadata',
                    str(metadata), '--output', str(output), *trial], cwd=ROOT, env=env, check=True)
    verify_episode(output, receipt, args.task)


if __name__ == '__main__':
    main()
