"""Defines dataset IO functions.
"""
from collections import namedtuple

import numpy as np


DataSet = namedtuple('DataSet', ['questions',  'sentences', 'labels'])


def question_batches(data_file):
    """Iterates over a dataset returning batches composed by a single question
    and its candidate answers.

    :data_file: a HDF5 file object holding the dataset
    :returns: a DataSet namedtuple of arrays (questions, sentences, labels).
    """
    n_questions = np.asscalar(data_file['metadata/questions/count'][...])
    questions_ds = data_file['data/questions']
    sentences_ds = data_file['data/sentences']

    for i in range(n_questions):
        row_labels = data_file['data/labels/q%d' % i][...]
        labels = row_labels[:, 1]
        rows = row_labels[:, 0]
        questions = questions_ds[rows, ...]
        sentences = sentences_ds[rows, ...]
        yield DataSet(questions, sentences, labels)


def question_epochs(data_file, n_epochs):
    """Iterates over a dataset for `n_epochs` epochs.
    """
    for i in range(n_epochs):
        yield from question_batches(data_file)


def trec_exporter(filepath):
    """Write TREC top file records to a text file.
    """
    scores = None
    qid = -1
    with open(filepath, 'w') as fout:
        while True:
            scores = yield scores
            qid += 1
            for sid in range(len(scores)):
                # writing TREC top file record
                print(qid, 0, sid, 0, '%.5f' % scores[sid], 'STANDARD',
                      file=fout)


def no_op(*args, **kwargs):
    """A convenience no-op coroutine to avoid `if is not None` checks.
    """
    dummy = None
    while True:
        dummy = yield dummy
