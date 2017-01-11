"""Prepare a dataset for the Answer Selection task. Selects only questions with
 at least one positive and one negative answer.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', help='original dev set (.jsonl)')
    args = parser.parse_args()
    infile = Path(args.data)
    outfile = Path(infile.stem + '-filtered' + infile.suffix)

    with infile.open() as fin,\
            outfile.open('w') as fout:
        for line in fin:
            row = json.loads(line)
            positive = sum(p['label'] for p in row['candidates'])
            negative = len(row['candidates']) - positive
            # at least one positive and one negative
            if positive > 0 and negative > 0:
                print(line.strip(), file=fout)


if __name__ == '__main__':
    main()
