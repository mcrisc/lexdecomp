"""Convert JSONL files to TSV.
"""
import argparse
import json
from pathlib import Path


HEADERS = ['QuestionID', 'Question', 'DocumentID', 'DocumentTitle',
           'SentenceID', 'Sentence', 'Label']


def main():
    parser = argparse.ArgumentParser(
            description='Convert JSONL files to TSV (compatible with '
                        'WikiQA)')
    parser.add_argument('data', help='data file')
    args = parser.parse_args()
    infile = Path(args.data)
    outfile = Path(infile.name).with_suffix('.tsv')

    with infile.open() as fin,\
            outfile.open('w') as fout:
        print('\t'.join(HEADERS), file=fout)
        for qid, line in enumerate(fin):
            row = json.loads(line)
            for i, p in enumerate(row['candidates']):
                print(qid, row['question'], 'D%d' % qid, 'doctitle',
                      i, p['sentence'], p['label'],
                      sep='\t', file=fout)


if __name__ == '__main__':
    main()
