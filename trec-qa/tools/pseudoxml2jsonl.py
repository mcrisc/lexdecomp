"""Convert TREC-QA pseudo-XML files to JSONL.
"""
import argparse
import json
import re
from pathlib import Path


RE_TAG = re.compile(r'^<(QApairs|positive|negative)')
RE_CLOSE = re.compile(r'^</QApairs>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', help='data set')
    args = parser.parse_args()

    fpath = Path(args.data)
    outfile = Path(fpath.name).with_suffix('.jsonl')

    row = {}
    with fpath.open() as fin,\
            outfile.open('w') as fout:
        while True:
            line = fin.readline()
            if line == '':  # EOF
                break
            if RE_CLOSE.search(line):
                print(json.dumps(row), file=fout)
            match = RE_TAG.search(line)
            if not match:
                continue
            tag = match.group(1)
            if tag == 'QApairs':
                fin.readline()  # discarding tag <question>
                line = fin.readline().strip()
                sentence = ' '.join(line.split('\t'))
                row = {'question': sentence, 'candidates': []}
            elif tag in ('positive', 'negative'):
                line = fin.readline().strip()
                sentence = ' '.join(line.split('\t'))
                label = 1 if tag == 'positive' else 0
                row['candidates'].append(
                        {'sentence': sentence, 'label': label})
    print('Output written to', outfile)


if __name__ == '__main__':
    main()
