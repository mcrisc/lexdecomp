import argparse
import re
import os
from pathlib import Path
from multiprocessing import Process

import numpy as np


# ensures Python 3.x
assert sys.version_info >= (3, 0)


RE_COORD = re.compile(r'-?\d+\.\d+')


def process_batch(data_file, dimension, start, batch_size):
    done = 0
    vocab_file = 'vocabulary-%05d.voc' % start
    matrix_file = 'matrix-%05d.npy' % start
    matrix = np.zeros((batch_size, dimension), dtype=np.float)
    with open(data_file) as fin, open(vocab_file, 'w') as fout:
        for i, line in enumerate(fin):
            if i < start:
                continue
            # begin job
            tokens = RE_COORD.findall(line)
            coords = tokens[-dimension:]
            word = line[:line.find(coords[0])]
            print(word, file=fout)
            vector = np.array([float(x) for x in coords], dtype=np.float)
            row = i - start
            matrix[row, :] = vector
            # end job
            done += 1
            if done == batch_size:  # finished batch
                break
    np.save(matrix_file, matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description='Export embeddings from text to binary '
                'NumPy arrays')
    parser.add_argument('source', help='embeddings text file')
    parser.add_argument('dimension', help='embedding dimension', type=int)
    args = parser.parse_args()
    source = Path(args.source)
    # output files
    vocab_file = source.with_suffix('.voc').name
    matrix_file = source.with_suffix('.npy').name
    dimension = args.dimension

    print('computing matrix dimensions')
    with source.open() as fin:
        line = next(fin)
        n_lines = sum((1 for _ in fin), 1)
    batch_size = n_lines // os.cpu_count()

    print('starting workers...')
    start = 0
    workers = []
    batches = []
    while start < n_lines:
        remaining = n_lines - start
        this_batch = batch_size if batch_size <= remaining else remaining
        p = Process(target=process_batch,
                    args=(args.source, dimension, start, this_batch))
        batches.append(start)
        p.start()
        workers.append(p)
        start += batch_size

    print('waiting...')
    for p in workers:
        p.join()

    print('concatenating vocabulary...')
    with open(vocab_file, 'w') as fout:
        for batch in batches:
            batch_file = Path('vocabulary-%05d.voc' % batch)
            # matrix_file = 'matrix-%05d.npy' % batch
            with batch_file.open() as fin:
                for line in fin:
                    print(line.strip(), file=fout)
            batch_file.unlink()

    print('concatenating partial matrices...')
    matrix = np.zeros((n_lines, dimension), dtype=np.float)
    i = 0
    for batch in batches:
        batch_file = Path('matrix-%05d.npy' % batch)
        partial = np.load(batch_file.as_posix())
        matrix[i: i+len(partial), :] = partial
        i += len(partial)
        batch_file.unlink()
    print('saving matrix...')
    np.save(matrix_file, matrix)
    print('finished')
