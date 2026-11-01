"""Package the two demos, source, data, and paper results."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'UCATE_Camera_Ready.zip')
    args = parser.parse_args()
    paths = [ROOT / n for n in ['README.md', 'requirements.txt', '.gitignore',
                               'UCATE_RealCause_Demo.ipynb', 'UCATE_Synthetic_Demo.ipynb']]
    for pattern in ['ours/*.py', 'scripts/*.py', 'tests/test_*.py', 'tests/fixtures/*.json',
                    'data/realcause/*', 'paper_results/README.md', 'paper_results/*/*']:
        paths.extend(p for p in ROOT.glob(pattern) if p.is_file())
    entries = {}
    for p in sorted(set(paths)):
        raw = p.read_bytes()
        if p.suffix == '.ipynb':
            notebook = json.loads(raw)
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    cell['outputs'], cell['execution_count'] = [], None
            raw = (json.dumps(notebook, ensure_ascii=False, indent=1) + '\n').encode()
        entries[p.relative_to(ROOT).as_posix()] = raw
    assert sum(n.endswith('.ipynb') for n in entries) == 2
    entries['MANIFEST.json'] = (json.dumps({k: hashlib.sha256(v).hexdigest()
                                           for k, v in entries.items()}, indent=2) + '\n').encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in entries.items():
            archive.writestr('UCATE_Camera_Ready/' + name, raw)
    with zipfile.ZipFile(args.output) as archive:
        assert archive.testzip() is None
    print(f'{args.output}: {len(entries)} files, {args.output.stat().st_size / 2**20:.1f} MiB')


if __name__ == '__main__':
    main()
