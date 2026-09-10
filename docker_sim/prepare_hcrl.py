"""Select source rows in [70%, 75%), without modifying CAN payload or IDS."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re

FILES = {'DoS': 'DoS_dataset.csv', 'Fuzzy': 'Fuzzy_dataset.csv',
         'RPM': 'RPM_dataset.csv', 'gear': 'gear_dataset.csv',
         'normal': 'normal_run_data/normal_run_data.txt'}

def parse(line, category):
    if category == 'normal':
        match = re.fullmatch(r'Timestamp:\s+(\S+)\s+ID:\s+(\S+)\s+\S+\s+DLC:\s+(\d+)\s*(.*)', line.strip())
        if not match: raise ValueError('normal log syntax')
        timestamp, cid, dlc, raw = match.groups()
        dlc = int(dlc); payload = bytes(int(x, 16) for x in raw.split())
        label = 'R'
    else:
        row = next(csv.reader([line])); timestamp, cid, dlc = row[:3]
        dlc = int(dlc); payload = bytes(int(x, 16) for x in row[3:3 + dlc])
        label = row[-1].strip()
    cid = int(cid, 16)
    if label not in ('T', 'R') or not 0 <= cid <= 0x7FF or not 0 <= dlc <= 8 or len(payload) != dlc:
        raise ValueError('invalid classic standard data frame')
    timestamp = float(timestamp)
    if not math.isfinite(timestamp): raise ValueError('invalid timestamp')
    return {'source_ts': timestamp, 'can_id': cid, 'dlc': dlc, 'data_hex': payload.hex(),
            'truth': category if label == 'T' else 'R'}

def main():
    p = argparse.ArgumentParser(); p.add_argument('dataset', type=Path)
    p.add_argument('--out', type=Path, default=Path(__file__).parent / 'prepared')
    args = p.parse_args(); args.out.mkdir(exist_ok=True, parents=True)
    manifest = {'selection': 'floor(N*0.7) through floor(N*0.7)+floor(N*0.05)-1 per file, zero based',
                'training_boundary': 'checked-in code uses first 70%; original training source hashes unverified', 'files': {}}
    for category, name in FILES.items():
        source = args.dataset / name; sha = hashlib.sha256(); count = 0
        with source.open('rb') as f:
            for line in f: sha.update(line); count += 1
        selected = count // 20; start = count * 7 // 10; labels = Counter(); times = []; collisions = 0
        target = args.out / (category + '.jsonl')
        with source.open() as src, target.open('w') as out:
            for index, line in enumerate(src):
                if index < start: continue
                if index >= start + selected: break
                row = parse(line, category)
                row.update(source_line=index + 1, index=index - start, dataset=category)
                if 0x701 <= row['can_id'] <= 0x705 and row['dlc'] == 8 and bytes.fromhex(row['data_hex'])[7] == 0xA5:
                    collisions += 1
                labels[row['truth']] += 1
                if not times: times.append(row['source_ts'])
                if len(times) == 1: times.append(row['source_ts'])
                else: times[-1] = row['source_ts']
                out.write(json.dumps(row, separators=(',', ':')) + '\n')
        if collisions: raise RuntimeError(f'{category}: synthetic application signature collides with dataset')
        info = {'source': str(source), 'sha256': sha.hexdigest(), 'total_rows': count,
                'selected_rows': selected, 'first_source_line': start + 1, 'last_source_line': start + selected,
                'selected_percent': selected / count * 100, 'label_counts': dict(labels),
                'source_span_seconds': times[-1] - times[0],
                'selected_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
        manifest['files'][category] = info
        print(category, json.dumps(info), flush=True)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))

if __name__ == '__main__': main()
