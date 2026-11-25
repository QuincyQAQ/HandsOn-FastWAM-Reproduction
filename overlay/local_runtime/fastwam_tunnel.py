"""Loopback-only HandsOn policy tunnel using the existing hkust SSH config."""
import argparse
import ast
import os
from pathlib import Path
import select
import socketserver
import paramiko


def connect():
    # Read just the existing helper's connection assignments. Never log secrets.
    source = Path(os.environ.get('HANDSON_SSH_CONFIG_HELPER', '/home/quincylee/.claude/scripts/hkust_ssh.py'))
    config = {}
    for node in ast.parse(source.read_text()).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name not in {'HOST', 'USER', 'PASS', 'PASSWORD'}:
            continue
        if isinstance(node.value, ast.Constant):
            config[name] = node.value.value
        elif (isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'get'
              and isinstance(node.value.func.value, ast.Attribute) and node.value.func.value.attr == 'environ'
              and isinstance(node.value.func.value.value, ast.Name) and node.value.func.value.value.id == 'os'):
            args = [ast.literal_eval(arg) for arg in node.value.args]
            config[name] = os.environ.get(*args)
    password = config.get('PASS') or config.get('PASSWORD')
    if not password:
        raise RuntimeError('Existing SSH helper has no supported password configuration')
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(config['HOST'], username=config['USER'], password=password,
                   look_for_keys=False, allow_agent=False, timeout=20)
    client.get_transport().set_keepalive(20)
    return client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8630)
    parser.add_argument('--metadata', type=Path, help='Fetch policy metadata, then exit')
    args = parser.parse_args()
    client = connect()
    try:
        if args.metadata:
            args.metadata.parent.mkdir(parents=True, exist_ok=True)
            with client.open_sftp() as sftp:
                sftp.get('/data1/qli736/robot/HandsOn-main/outputs/fastwam-cup/policy-metadata.json', str(args.metadata))
            print('Policy metadata fetched.', flush=True)
            return
        transport = client.get_transport()

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                channel = transport.open_channel('direct-tcpip', ('127.0.0.1', args.port), self.client_address)
                if channel is None:
                    return
                try:
                    while True:
                        ready, _, _ = select.select([self.request, channel], [], [], 30)
                        if not ready and not transport.is_active():
                            return
                        for source in ready:
                            data = source.recv(65536)
                            if not data:
                                return
                            (channel if source is self.request else self.request).sendall(data)
                finally:
                    channel.close()

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        with Server(('127.0.0.1', args.port), Handler) as server:
            print(f'HandsOn tunnel ready: 127.0.0.1:{args.port} -> hkust loopback:{args.port}', flush=True)
            while transport.is_active():
                server.timeout = 5
                server.handle_request()
            raise ConnectionError('SSH transport closed')
    finally:
        client.close()


if __name__ == '__main__':
    main()
