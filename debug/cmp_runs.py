import csv

RUNS = {
    'off': r'e:\gitProjects\OSTrack\output\batch_eval\uni0_refpool0_v_0727_140402\metrics.csv',
    'tcm_v4': r'e:\gitProjects\OSTrack\output\batch_eval\tcm_v4_full\metrics.csv',
    'tcm_v5': r'e:\gitProjects\OSTrack\output\batch_eval\tcm_v5g_full\metrics.csv',
}

data = {k: {r['sequence']: r for r in csv.DictReader(open(p))} for k, p in RUNS.items()}
seqs = [s for s in data['off'] if all(s in data[k] and data[k][s]['success'] for k in RUNS)]

print('TCM-v5 (distractor bank) vs TCM-v4 (only |dSuccess|>0.01):')
diffs = []
for s in seqs:
    d = float(data['tcm_v5'][s]['success']) - float(data['tcm_v4'][s]['success'])
    if abs(d) > 0.01:
        diffs.append((d, s))
for d, s in sorted(diffs):
    v4_off = float(data['tcm_v4'][s]['success']) - float(data['off'][s]['success'])
    print(f'{s:<16} v5-v4: {d:+.3f}   (v4-off was {v4_off:+.3f})')
