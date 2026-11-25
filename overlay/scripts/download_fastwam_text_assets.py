"""User-started, resumable download of only the frozen encoder/tokenizer."""
import hashlib
from pathlib import Path
import subprocess

ROOT = Path('/data1/qli736/robot/weights/handson/wan-text')
REVISION = '921dbaf3f1674a56f47e83fb80a34bac8a8f203e'
FILES = [
 ('models_t5_umt5-xxl-enc-bf16.pth',11361920418,'sha256','7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d'),
 ('google/umt5-xxl/spiece.model',4548313,'sha256','e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458'),
 ('google/umt5-xxl/tokenizer.json',16837417,'sha256','6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b'),
 ('google/umt5-xxl/special_tokens_map.json',6623,'git','14855e7052ffbb595057dfd791d293c1c940db2c'),
 ('google/umt5-xxl/tokenizer_config.json',61728,'git','4e1cc1cd85599ce0b47fd0a746af188fe4043ff2'),
]

def verified(path, size, kind, expected):
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = hashlib.sha256() if kind == 'sha256' else hashlib.sha1()
    if kind == 'git':
        digest.update(f'blob {size}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest() == expected

def main():
    for name, size, kind, expected in FILES:
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if verified(path, size, kind, expected):
            print(f'Already verified: {name}', flush=True)
            continue
        if path.exists():
            raise RuntimeError(f'Existing final file failed validation: {path}; inspect it before retrying')
        temporary = path.with_name(path.name + '.part')
        if temporary.exists() and temporary.stat().st_size == size:
            if verified(temporary, size, kind, expected):
                temporary.replace(path)
                continue
            raise RuntimeError(f'Complete partial file has a bad hash: {temporary}')
        print(f'Downloading {name}: {size:,} bytes', flush=True)
        subprocess.run(['curl','--http1.1','--tlsv1.2','--tls-max','1.2',
            '--fail','--location','--continue-at','-','--retry','20',
            '--retry-all-errors','--retry-delay','5','--connect-timeout','20',
            '--output',str(temporary),
            f'https://hf-mirror.com/Wan-AI/Wan2.2-TI2V-5B/resolve/{REVISION}/{name}'], check=True)
        if not verified(temporary, size, kind, expected):
            raise RuntimeError(f'File failed size/hash validation: {temporary}')
        temporary.replace(path)
        print(f'Verified: {name}', flush=True)
    print('All FastWAM text assets verified. No policy rollout has been run.', flush=True)

if __name__ == '__main__':
    main()
