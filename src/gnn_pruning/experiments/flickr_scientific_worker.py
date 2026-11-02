"""Scientific-only Flickr resource bootstrap; affinity is applied before Torch import.

This entry point is used solely by the56 authorized independent scientific jobs.
No technical, smoke, profiling, fitting or postprocessing mode is available.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'protocol', 'resources'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--condition', required=True)
    args = parser.parse_args()
    profiles = json.loads(args.resources.read_text())
    profile = dict(profiles[args.condition])
    if profile['threads'] == 12:
        expected = 'flickr-graphsage-l4-h256-s101-tabular_q-t'
        cpus = profile['cpu_affinity']
        if (not args.condition.startswith(expected) or profile['interop_threads'] != 1
            or len(cpus) != 20 or len(set(cpus)) != 20):
            raise ValueError('Unknown high-thread resource profile.')
        if not set(cpus) <= os.sched_getaffinity(0):
            raise RuntimeError('Allocation excludes the verified physical-CPU mask.')
        os.environ.pop('GOMP_CPU_AFFINITY', None)
        os.sched_setaffinity(0, cpus)
        if os.sched_getaffinity(0) != set(cpus):
            raise RuntimeError('Verified physical-CPU affinity could not be applied.')
    elif profile['threads'] != 4 or profile.get('cpu_affinity') is not None:
        raise ValueError('Unknown four-thread resource profile.')
    for name, value in profile['thread_environment'].items():
        os.environ[name] = value
    profile['observed_affinity'] = sorted(os.sched_getaffinity(0))
    # Only authorized scientific workers reach the Torch-dependent implementation.
    from gnn_pruning.experiments.flickr_pruning import execute, matrix
    protocol = json.loads(args.protocol.read_text())
    condition = next(c for c in matrix(protocol) if c['id'] == args.condition)
    execute(args.root, protocol, condition, technical=False, execution_resources=profile)


if __name__ == '__main__':
    main()
