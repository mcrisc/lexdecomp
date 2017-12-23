import argparse
import mmap
import os
import struct
import sys
from array import array
from contextlib import contextmanager

# ensures Python 3.x
assert sys.version_info >= (3, 0)


@contextmanager
def memorymap(filename):
    try:
        size = os.path.getsize(filename)
        fd = os.open(filename, os.O_RDONLY)
        mapped_file = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
        yield mapped_file
    finally:
        mapped_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description='Export embeddings from binary word2vec files '
                'to plain text')
    parser.add_argument('source', help='word2vec binary file')
    parser.add_argument('dest', help='destination file')
    parser.add_argument('--vocabulary', help='text file containing words to '
                        'export (one word per line)')
    args = parser.parse_args()

    if args.vocabulary:
        print('loading vocabulary')
        with open(args.vocabulary) as fin:
            vocabulary = {word.strip() for word in fin}
    else:
        vocabulary = None

    print('exporting vectors')
    with memorymap(args.source) as mvec, \
            open(args.dest, 'w') as fout:
        end = mvec.find(b'\n', 0, 100)
        if end == -1:
            raise Exception("Invalid file format")
        _, token2 = mvec[0:end].split()
        vector_size = int(token2)
        byte_offset = vector_size * struct.calcsize('f')
        while True:
            # reading a word
            pos = end
            if pos >= mvec.size():
                break
            end = mvec.find(b' ', pos)
            if end == -1:
                break
            wordbytes = mvec[pos:end]
            try:
                word = wordbytes.decode('utf-8', errors='replace').strip()
                # reading the corresponding vector
                pos = end + 1
                end = pos + byte_offset
                vector = array('f', mvec[pos:end])
                if vocabulary is not None and word not in vocabulary:
                    continue  # skip word if not in vocabulary
                else:
                    print(word, ' '.join(map(str, vector)), file=fout)
            except:
                continue
        print('finished')
