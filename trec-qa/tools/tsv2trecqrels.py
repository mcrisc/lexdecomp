"""Generates a TREC relevance file (qrel or judgement) from a WikiQA data file.
"""
import argparse
from collections import namedtuple
from pathlib import Path


WikiQARow = namedtuple(
    'WikiQARow',
    ['question_id', 'question', 'document_id', 'document_title', 'sentence_id',
     'sentence', 'label'])


def main():
    parser = argparse.ArgumentParser(
        description='Generates a TREC .ref (qrel) file')
    parser.add_argument('data', help='dataset file in TSV format (WikiQA)')
    args = parser.parse_args()

    data_file = Path(args.data)
    out_file = Path(data_file.name).with_suffix('.qrels')

    with data_file.open() as fin,\
            out_file.open('w') as fout:
        next(fin)  # skipping headers
        for line in fin:
            row = WikiQARow._make(line.strip().split('\t'))
            print(row.question_id, '0', row.sentence_id, row.label,
                  sep='\t', file=fout)

    print('Output written to', out_file)


if __name__ == '__main__':
    main()
