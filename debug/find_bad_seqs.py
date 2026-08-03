"""List GOT-10k val sequences worth visual inspection:

  A) regressions : improved AO dropped more than 10% (relative) vs origin
  B) both-weak   : origin AO AND improved AO both below 0.5

Usage:
    python debug/find_bad_seqs.py [origin_tag improved_tag]
Defaults: got10k_val_origin got10k_val_dtv2_noprior
"""
import sys

from cmp_got10k import load


def main():
    origin_tag = sys.argv[1] if len(sys.argv) > 2 else 'got10k_val_origin'
    imp_tag = sys.argv[2] if len(sys.argv) > 2 else 'got10k_val_dtv2_noprior'
    origin = load(origin_tag)
    imp = load(imp_tag)
    common = sorted(set(origin) & set(imp))

    regress = []   # (seq, origin_ao, imp_ao, rel_drop)
    weak = []      # (seq, origin_ao, imp_ao)
    for s in common:
        o, n = origin[s], imp[s]
        if o > 0 and (o - n) / o > 0.10:
            regress.append((s, o, n, (o - n) / o))
        if o < 0.5 and n < 0.5:
            weak.append((s, o, n))

    print(f'=== A) Regressions >10% relative ({len(regress)} seqs) ===')
    print(f'{"seq":<24}{"origin":>10}{"improved":>10}{"drop":>9}')
    for s, o, n, d in sorted(regress, key=lambda x: -x[3]):
        print(f'{s:<24}{o:>10.3f}{n:>10.3f}{d:>8.1%}')

    print(f'\n=== B) Both weak (<0.5) ({len(weak)} seqs) ===')
    print(f'{"seq":<24}{"origin":>10}{"improved":>10}')
    for s, o, n in sorted(weak, key=lambda x: x[2]):
        print(f'{s:<24}{o:>10.3f}{n:>10.3f}')

    both = sorted(set(x[0] for x in regress) | set(x[0] for x in weak))
    print(f'\n=== Union for inspection ({len(both)} seqs) ===')
    print(','.join(both))


if __name__ == '__main__':
    main()
