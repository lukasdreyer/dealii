#!/usr/bin/env python3
"""Summarize the CSV files of run_comparison.sh: for every backend and number
of processes, the sum over the cycles of the maximal (over the processes) time
of each phase, the minimum over the repetitions."""
import csv, glob, os, re, sys
from collections import defaultdict

out = sys.argv[1] if len(sys.argv) > 1 else 'results'
totals = defaultdict(lambda: defaultdict(lambda: float('inf')))
phases = []
for path in sorted(glob.glob(os.path.join(out, '*_np*_rep*.csv'))):
    m = re.match(r'(.*)_np(\d+)_rep(\d+)\.csv', os.path.basename(path))
    key = (m.group(1), int(m.group(2)))
    sums = defaultdict(float)
    with open(path) as f:
        for row in csv.DictReader(f):
            sums[row['phase']] += float(row['max'])
            if row['phase'] not in phases:
                phases.append(row['phase'])
    sums['total'] = sum(sums.values())
    for phase, t in sums.items():
        totals[key][phase] = min(totals[key][phase], t)

keys = sorted(totals, key=lambda k: (k[1], k[0]))
print('%-18s' % 'phase [s]' + ''.join('%14s' % ('%s np%d' % k) for k in keys))
for phase in phases + ['total']:
    print('%-18s' % phase + ''.join('%14.3f' % totals[k][phase] for k in keys))
