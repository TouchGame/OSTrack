import csv

RUNS = {
    'off': r'e:\gitProjects\OSTrack\output\batch_eval\uni0_refpool0_v_0727_140402\metrics.csv',
    'tcm_v1': r'e:\gitProjects\OSTrack\output\batch_eval\uni0_refpool1_v\metrics.csv',
    'tcm_v4': r'e:\gitProjects\OSTrack\output\batch_eval\tcm_v4_full\metrics.csv',
}

data = {k: {r['sequence']: r for r in csv.DictReader(open(p))} for k, p in RUNS.items()}
seqs = [s for s in data['off'] if all(s in data[k] and data[k][s]['success'] for k in RUNS)]

print('TCM-v4 vs all-off (only |dSuccess|>0.01):')
diffs = []
for s in seqs:
    d = float(data['tcm_v4'][s]['success']) - float(data['off'][s]['success'])
    if abs(d) > 0.01:
        diffs.append((d, s))
for d, s in sorted(diffs):
    v1 = float(data['tcm_v1'][s]['success']) - float(data['off'][s]['success'])
    print(f'{s:<16} v4: {d:+.3f}   (v1 was {v1:+.3f})')
