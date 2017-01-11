"""Prepare text that will be further represented in distributed vector space.
"""

import argparse
import logging
import re

from collections import namedtuple
from pathlib import Path
from nltk.tokenize import word_tokenize


RE_DIGITS = re.compile(r'\d')
Record = namedtuple('Record', ['question_id', 'question', 'document_id',
                               'document_title', 'sentence_id', 'sentence',
                               'label'])


def tokenize(sentence):
    """Tokenize sentences. Same method used for embedding generation."""
    tokens = [RE_DIGITS.sub('#', w) for w in word_tokenize(sentence)]
    tokenized = ' '.join(tokens)
    return tokenized


def main():
    parser = argparse.ArgumentParser(
        description='Prepare text that will be further represented in '
                    'distributed vector space.')
    parser.add_argument('data', help='dataset file in TSV format (WikiQA)')
    parser.add_argument(
        '-f', '--filter',
        help='file containing question-sentence relations in TREC .ref format')
    args = parser.parse_args()

    infile = Path(args.data)
    outfile = Path(infile.name).with_suffix('.txt')

    filtered = None
    if args.filter:
        with Path(args.filter).open() as fin:
            filtered = {int(line[:line.find(' ')]) for line in fin}

    logging.info('Processing %s' % infile)
    with infile.open() as fin, \
            outfile.open('w') as fout:
        next(fin)  # skipping headers
        qid = 'no-qid'
        current_question = 0
        for line in fin:
            record = Record._make(line.strip().split('\t'))
            if record.question_id != qid:
                qid = record.question_id
                current_question += 1
                question = tokenize(record.question)
            if filtered and current_question not in filtered:
                continue  # skipping filtered questions
            sentence = tokenize(record.sentence)
            print(current_question, question, sentence, record.label,
                  sep='\t', file=fout)
    logging.info('Output written to %s' % outfile)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')
    main()
