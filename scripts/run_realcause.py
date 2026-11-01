"""Run the RealCause demo from the command line."""
import argparse
from pathlib import Path
import sys
import json
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ours.realcause_experiments import experiment_config, run_samples, summarize_results, make_run_manifest, save_manifest
from ours.realcause_reporting import analyze_saved_run, save_noise_report, plot_noise_comparison
from ours.demo import realcause_run_dir


def run_file(config):
    return run_samples(config, verbose=False).iloc[0].to_dict()


def run_parallel(config, out, workers):
    """Independent files; write an ordered, resumable prefix after each result."""
    out.mkdir(parents=True, exist_ok=True)
    manifest = make_run_manifest(config)
    checkpoint = out / 'samples.csv'
    rows = []
    if checkpoint.exists():
        previous = json.loads((out / 'manifest.json').read_text())
        for key in ['config', 'source_sha256', 'dependencies']:
            if previous[key] != manifest[key]:
                raise ValueError(f'Checkpoint {key} differs; use a new output directory.')
        rows = pd.read_csv(checkpoint).to_dict('records')
        if [int(r['sample']) for r in rows] != config['samples'][:len(rows)]:
            raise ValueError('Checkpoint must be an ordered prefix of this run.')
        if any(r['backend'] != 'nystrom' for r in rows):
            raise ValueError('Checkpoint is not a Nyström run.')
    else:
        manifest['file_workers'] = workers
        save_manifest(out / 'manifest.json', manifest)
    pending = [dict(config, samples=[i]) for i in config['samples'][len(rows):]]
    if pending:
        saved = json.loads((out / 'manifest.json').read_text())
        saved.setdefault('execution_segments', []).append(
            {'first_pending_sample': pending[0]['samples'][0], 'file_workers': workers})
        save_manifest(out / 'manifest.json', saved)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(run_file, pending):
            rows.append(row)
            temporary = checkpoint.with_suffix('.tmp')
            pd.DataFrame(rows).to_csv(temporary, index=False)
            temporary.replace(checkpoint)
            print(f"[{len(rows)}/{len(config['samples'])}] sample {row['sample']}: "
                  f"Ours={row['mse_ours']:.6f}, {row['seconds_total']:.1f}s", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-samples', type=int, default=50)
    parser.add_argument('--start-sample', type=int, default=0)
    parser.add_argument('--profile', choices=['quick', 'full'], default='full')
    parser.add_argument('--workers', type=int, default=1, help='Independent dataset files in parallel.')
    args = parser.parse_args()
    config = experiment_config(args.profile, args.n_samples, args.start_sample)
    out = realcause_run_dir(config, ROOT / 'outputs' / 'realcause')
    if args.workers < 1:
        parser.error('--workers must be positive')
    results = (run_samples(config, checkpoint_path=out / 'samples.csv') if args.workers == 1
               else run_parallel(config, out, args.workers))
    summary = summarize_results(results)
    summary.to_csv(out / 'summary.csv', index=False)
    print(summary.to_string(index=False))
    if args.profile == 'full' and len(results) >= 3:
        report = analyze_saved_run(results)
        save_noise_report(report, out)
        fig = plot_noise_comparison(report['summary'])
        fig.savefig(out / 'comparison.pdf', bbox_inches='tight')
        fig.savefig(out / 'comparison.png', dpi=180, bbox_inches='tight')
        print(report['summary'].to_string(index=False))


if __name__ == '__main__':
    main()
