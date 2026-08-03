"""Pack a GOT-10k test submission zip, wrapped in a top-level tracker folder.

Layout produced (all forward slashes, as the GOT-10k server expects):
    <tracker_name>/<seq>/<seq>_001.txt
    <tracker_name>/<seq>/<seq>_time.txt

NEVER use PowerShell Compress-Archive for this (it writes backslash entry
names that the Linux-side evaluator cannot unpack).

Usage:
    python pack_submission.py <submission_dir> <tracker_name> [out_zip]

After writing, the zip is re-opened and verified:
  * 180 sequence folders x 2 files = 360 entries expected
  * every entry name starts with '<tracker_name>/' and contains no '\\'
  * first line of GOT-10k_Test_000001_001.txt is echoed for a GT sanity check
"""
import os
import sys
import zipfile


def main():
    sub_dir = sys.argv[1]
    tracker = sys.argv[2]
    out_zip = sys.argv[3] if len(sys.argv) > 3 else \
        os.path.join(os.path.dirname(sub_dir.rstrip('/\\')), tracker + '.zip')

    seqs = sorted(d for d in os.listdir(sub_dir)
                  if os.path.isdir(os.path.join(sub_dir, d)))
    if not seqs:
        sys.exit(f'no sequence folders under {sub_dir}')

    n_files = 0
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for seq in seqs:
            seq_dir = os.path.join(sub_dir, seq)
            for fname in sorted(os.listdir(seq_dir)):
                # arcname built with '/' explicitly: never trust os.path here
                zf.write(os.path.join(seq_dir, fname),
                         arcname=f'{tracker}/{seq}/{fname}')
                n_files += 1
    print(f'Wrote {out_zip}: {len(seqs)} sequences, {n_files} entries')

    # ── verification pass ──
    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
        bad_prefix = [n for n in names if not n.startswith(tracker + '/')]
        bad_slash = [n for n in names if '\\' in n]
        assert not bad_prefix, f'entries outside wrapper: {bad_prefix[:3]}'
        assert not bad_slash, f'backslash entries: {bad_slash[:3]}'
        probe = f'{tracker}/GOT-10k_Test_000001/GOT-10k_Test_000001_001.txt'
        if probe in names:
            first = zf.read(probe).decode().splitlines()[0]
            print(f'Verify OK: {len(names)} entries, all under "{tracker}/", '
                  f'no backslashes.\n000001 first box: {first} '
                  f'(GT: 395.00,340.00,532.00,407.00)')
        else:
            print(f'Verify OK: {len(names)} entries, all under "{tracker}/", '
                  f'no backslashes. (probe seq not in this zip)')


if __name__ == '__main__':
    main()
