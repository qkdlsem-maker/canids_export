"""Local-only synthetic CAN laboratory; the MCU is the IDS decision authority."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent
STOP = threading.Event()
LOCK = threading.Lock()
EVENTS = []
GATE = {'confirmed': 0, 'errors': 0, 'modes': set()}
RUN = None
BUS = None
LOG_FILE = None
MATCH = threading.Condition()
PENDING = None
REPLAY_ABORT = None
LAST_SEND = {}
READER_READY = [threading.Event(), threading.Event()]
FIELDS = ['sequence', 'can_id', 'dlc', 'low', 'high', 'prediction', 'rx_tick', 'end_tick']
SL_IDS = {0x0D5, 0x0EF, 0x0D2, 0x1D2, 0x2D2}
STARTED = set()
PROTOCOL_ERROR = False

def emit(kind, **fields):
    global PROTOCOL_ERROR
    item = dict(kind=kind, host_monotonic_ns=time.monotonic_ns(), **fields)
    with LOCK:
        LOG_FILE.write(json.dumps(item, separators=(',', ':')) + '\n')
        EVENTS.append(item)
        if kind == 'firmware':
            event = fields['event']
            if event == 'K': GATE['confirmed'] += 1
            if event in ('O', 'M', 'D', 'E'): GATE['errors'] += 1
            if event == 'Z' and any(fields[k] for k in ('dlc', 'low', 'high', 'end_tick')):
                GATE['errors'] += 1
            if event == 'S' and fields['high'] != 2:
                GATE['errors'] += 1
                PROTOCOL_ERROR = True
            if event == 'S':
                STARTED.add(fields['ecu'])
            if event == 'K' and fields['ecu'] == 'A' and fields['can_id'] == 0x2D2:
                GATE['modes'].add((fields['low'] >> 16) & 255)
        if len(EVENTS) > 1000:
            del EVENTS[:500]

def uart_reader(index):
    try:
        for _ in range(100):
            try:
                conn = socket.create_connection(('127.0.0.1', 7001 + index), 0.3)
                break
            except OSError:
                if STOP.wait(0.1): return
        else:
            raise RuntimeError('UART connection timed out')
        conn.settimeout(None)
        with conn, conn.makefile('rb') as stream:
            READER_READY[index].set()
            for raw in stream:
                line = raw.decode('ascii', errors='replace').strip()
                parts = line.split(',')
                if len(parts) == 10 and parts[0] in ('A', 'B'):
                    try:
                        fields = dict(zip(FIELDS, (int(x, 16) for x in parts[2:])))
                    except ValueError:
                        emit('uart_unparsed', ecu=index, raw=line)
                        continue
                    if parts[1] != 'Z':
                        fields['data_hex'] = struct.pack('<II', fields['low'], fields['high']).hex()[:fields['dlc'] * 2]
                    global PENDING
                    with MATCH:
                        matched = (parts[0] == 'B' and parts[1] == 'R' and PENDING is not None and
                            (fields['can_id'], fields['dlc'], fields.get('data_hex')) ==
                            (PENDING['can_id'], PENDING['dlc'], PENDING['data_hex']))
                        if matched:
                            fields['dataset'] = PENDING['dataset']
                            fields['source_index'] = PENDING['index']
                            fields['truth'] = PENDING['truth']
                            fields['submit_to_log_ms'] = (time.monotonic_ns() - PENDING['submit_ns']) / 1e6
                        emit('firmware', ecu=parts[0], event=parts[1], **fields)
                        if matched:
                            PENDING = None
                            MATCH.notify_all()
                elif line:
                    emit('uart_unparsed', ecu=index, raw=line)
    except Exception as exc:
        emit('reader_error', ecu=index, error=str(exc))
    finally:
        READER_READY[index].clear()

def bus_reader():
    while not STOP.is_set():
        try:
            raw = BUS.recv(16)
        except socket.timeout:
            continue
        cid, dlc, payload = struct.unpack('=IB3x8s', raw)
        emit('bus_frame', can_id=cid, dlc=dlc, data_hex=payload[:dlc].hex())

def replay_dataset(category):
    """Backpressured replay to this container's isolated vcan only.

    Preserves frame order and bytes; not original-rate bus-load validation.
    Labels stay in the host log and are never passed to the firmware.
    """
    global PENDING, REPLAY_ABORT
    try:
        source = ROOT / 'prepared' / (category + '.jsonl')
        emit('replay_start', dataset=category, selected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             timing='wait for B decision before next source frame; no original-rate claim')
        started = time.monotonic()
        with source.open() as f:
            for line in f:
                if STOP.is_set(): raise RuntimeError('stopped before replay completed')
                row = json.loads(line)
                with MATCH:
                    row['submit_ns'] = time.monotonic_ns()
                    PENDING = row
                    emit('dataset_submit', **row)
                    BUS.send(struct.pack('=IB3x8s', row['can_id'], row['dlc'], bytes.fromhex(row['data_hex'])))
                    if not MATCH.wait_for(lambda: PENDING is None or STOP.is_set(), timeout=10):
                        raise RuntimeError(f"B frame confirmation timeout at row {row['index']}")
                    if STOP.is_set() or PENDING is not None:
                        raise RuntimeError('interrupted before B confirmation')
                if (row['index'] + 1) % 10000 == 0:
                    LOG_FILE.flush()
                    print(f"{category}: {row['index'] + 1} frames, {(row['index'] + 1)/(time.monotonic()-started):.1f} fps", flush=True)
        emit('replay_complete', dataset=category, seconds=time.monotonic() - started)
    except Exception as exc:
        REPLAY_ABORT = str(exc)
        emit('replay_failed', dataset=category, error=REPLAY_ABORT)
    finally:
        LOG_FILE.flush()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def reply(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/health':
            self.reply(200, {'ready': all(e.is_set() for e in READER_READY), 'run': RUN.name})
        elif self.path == '/events':
            with LOCK: events = list(EVENTS)
            self.reply(200, events)
        else:
            self.reply(404, {'error': 'not found'})

    def do_POST(self):
        if self.path != '/fixture':
            return self.reply(404, {'error': 'not found'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 1 <= size <= 1024:
                raise ValueError('body size must be 1..1024')
            value = json.loads(self.rfile.read(size))
            role, seq = value['scenario'], value['sequence']
            if role not in ('normal', 'anomaly') or type(seq) is not int or not 0 <= seq < 2**32:
                raise ValueError('invalid fixture')
            # Two bounded synthetic fixtures, not arbitrary CAN injection.
            cid = 0x610 if role == 'normal' else 0x611
            payload = struct.pack('<I', seq) + (b'\x10\x20\x30\x40' if role == 'normal' else b'\xff' * 4)
            with LOCK:
                now = time.monotonic()
                if now - LAST_SEND.get(role, -1) < 0.08:
                    return self.reply(429, {'error': 'fixture limit: 12.5 frames/s'})
                LAST_SEND[role] = now
            BUS.send(struct.pack('=IB3x8s', cid, 8, payload))
            emit('fixture_submitted', scenario=role, source_sequence=seq,
                 can_id=cid, data_hex=payload.hex())
            self.reply(202, {'bus_submitted': True, 'ecu_received': 'check firmware log', 'can_id': cid})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.reply(400, {'error': str(exc)})

def main():
    global RUN, BUS, LOG_FILE
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-vcan', action='store_true', help='only A/B firmware test, no HTTP input')
    parser.add_argument('--duration', type=float, default=0)
    parser.add_argument('--dataset', choices=['DoS', 'Fuzzy', 'RPM', 'gear', 'normal'])
    parser.add_argument('--baseline-seconds', type=float, default=12)
    args = parser.parse_args()
    if args.dataset and args.no_vcan:
        parser.error('--dataset requires the isolated vcan bridge')
    if args.duration < 0 or args.baseline_seconds < 0:
        parser.error('duration and baseline-seconds must be nonnegative')
    if args.dataset:
        # A/B have no source tags on the CAN wire. Refuse ambiguous attribution
        # and command interference; never silently filter or relabel dataset rows.
        with (ROOT / 'prepared' / (args.dataset + '.jsonl')).open() as source:
            for line in source:
                row = json.loads(line)
                if row['can_id'] in SL_IDS:
                    raise ValueError('Dataset overlaps SL application IDs; mixed replay requires '
                                     'independent source attribution. No frames sent or removed.')
    os.chdir(ROOT)
    RUN = ROOT / 'results' / (time.strftime('%Y%m%d-%H%M%S') + '-' + (args.dataset or 'baseline'))
    RUN.mkdir(parents=True)
    LOG_FILE = (RUN / 'events.jsonl').open('a', buffering=262144)
    hashes = {}
    for path in [ROOT / 'firmware.c', ROOT / 'sl_application.h', ROOT / 'service.py',
                 *sorted((ROOT / 'build').glob('*.elf')),
                 ROOT / '../models_c/can_ids_embedded_v2.c', ROOT / '../models_c/f6_v2/feature_v2_stream.c']:
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (RUN / 'manifest.json').write_text(json.dumps({'hashes': hashes, 'protocol': 'sl-functional-v1',
        'scope': 'Renode functional simulation; ticks are virtual 10ms units; instrumentation included',
        'assumptions': 'SL_PROTOCOL.md: HVTGT offset +250 V; CHKGRP=0 unverified; not HCRL normal ground truth',
        'fixtures': 'synthetic scenario labels, not validated HCRL ground truth'}, indent=2))
    if args.dataset:
        (RUN / 'selection_manifest.json').write_bytes((ROOT / 'prepared/manifest.json').read_bytes())
    if not args.no_vcan:
        BUS = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        BUS.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_RECV_OWN_MSGS, 1)
        BUS.bind(('vcanjarvis',))
        BUS.settimeout(0.5)
    env = dict(os.environ, XDG_CONFIG_HOME='/tmp/jarvis-renode-config')
    commands = 'include @two_ecus.resc'
    if not args.no_vcan: commands += '; include @socketcan.resc'
    log = (RUN / 'renode.log').open('w')
    proc = subprocess.Popen([str(ROOT / '../models_c/renode_portable/renode'), '--disable-xwt',
        '--plain', '--port', '7000', '-e', commands], env=env, stdout=log, stderr=subprocess.STDOUT)
    server = None
    readers = []
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    try:
        for i in range(2):
            thread = threading.Thread(target=uart_reader, args=(i,), daemon=True)
            readers.append(thread)
            thread.start()
        if not all(event.wait(15) for event in READER_READY):
            raise RuntimeError('Renode UART initialization failed; inspect renode.log')
        monitor = socket.create_connection(('127.0.0.1', 7000), 5)
        monitor.sendall(b'start\n')
        time.sleep(0.1)
        if BUS:
            threading.Thread(target=bus_reader, daemon=True).start()
            # Legacy synthetic HTTP fixtures are not SL-normal examples.
        print(f'JARVIS simulation ready; results={RUN}', flush=True)
        start = time.monotonic()
        replay = None
        while not STOP.wait(0.2):
            if proc.poll() is not None: raise RuntimeError('Renode exited unexpectedly')
            if not all(e.is_set() for e in READER_READY): raise RuntimeError('UART logger disconnected')
            with LOCK:
                if PROTOCOL_ERROR:
                    raise RuntimeError('Legacy firmware startup marker; rebuild both ECU images')
                startup_complete = STARTED == {'A', 'B'}
                if not startup_complete and time.monotonic() - start > 15:
                    raise RuntimeError('Both SL firmware startup markers were not received')
            if args.dataset and replay is None and args.duration and time.monotonic() - start >= args.duration:
                raise RuntimeError('Duration expired before dataset replay could start')
            if args.dataset and startup_complete and replay is None and time.monotonic() - start >= args.baseline_seconds:
                with LOCK: gate = dict(GATE)
                if not gate['confirmed']:
                    raise RuntimeError('normal communication gate failed: no confirmed response')
                if gate['errors']:
                    raise RuntimeError('normal communication gate failed: application error')
                if len(gate['modes']) < 4:
                    if time.monotonic() - start > max(120, args.baseline_seconds):
                        raise RuntimeError('normal communication gate timed out: four matching states absent')
                    continue
                emit('normal_gate_pass', dataset=args.dataset)
                replay = threading.Thread(target=replay_dataset, args=(args.dataset,))
                replay.start()
            if replay is not None and not replay.is_alive():
                if REPLAY_ABORT: raise RuntimeError(REPLAY_ABORT)
                time.sleep(0.5)
                break
            if args.duration and time.monotonic() - start >= args.duration:
                if not startup_complete:
                    raise RuntimeError('Duration expired before both SL firmware startup markers')
                if args.dataset:
                    raise RuntimeError('Duration expired before complete dataset replay')
                break
    finally:
        emit('shutdown_boundary', note='Cross-layer completeness applies only before this boundary; final partial exchanges may remain.')
        STOP.set()
        with MATCH: MATCH.notify_all()
        if 'replay' in locals() and replay is not None: replay.join(timeout=12)
        if server: server.shutdown()
        proc.terminate()
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        for thread in readers: thread.join(timeout=3)
        if 'monitor' in locals(): monitor.close()
        log.close()
        LOG_FILE.flush()
        LOG_FILE.close()

if __name__ == '__main__': main()
