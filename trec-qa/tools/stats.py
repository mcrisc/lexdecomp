import argparse
import json
import math
import re

import numpy as np
from scipy import stats


RE_TOKENS = re.compile(r'\S+')
PERCENTILES = np.array([1, 5, 10, 25, 50, 75, 88, 90, 95, 99, 99.95])


def tokens(text):
    return len(RE_TOKENS.findall(text))


def print_stats(data):
    data = np.array(data)
    desc = stats.describe(data)
    print('# of observations:', desc.nobs)
    print('min: %d\nmax: %d' % desc.minmax)
    print('mean: %.1f' % desc.mean)
    # print('variance: %.1f' % desc.variance)
    print('stdev: %.1f' % math.sqrt(desc.variance))

    print('percentiles')
    for p in PERCENTILES:
        print('%6.2f' % p, '  ', end='')
    print()
    for p in stats.scoreatpercentile(data, PERCENTILES):
        print('%6d' % p, '  ', end='')
    print()


def main():
    parser = argparse.ArgumentParser(
            description='Compute stats from JSONL files')
    parser.add_argument('file', help='data file')
    args = parser.parse_args()

    query_lengths = []
    correct_answers = []
    candidates_count = []
    candidate_lengths = []
    with open(args.file) as fin:
        for line in fin:
            row = json.loads(line)
            query_lengths.append(tokens(row['question']))
            correct_answers.append(sum(1 for p in row['candidates']
                                   if p['label'] == 1))
            candidates_count.append(len(row['candidates']))
            candidate_lengths.extend([tokens(p['sentence'])
                                      for p in row['candidates']])

    statistics = [('Question Lengths', query_lengths),
                  ('Candidate Length', candidate_lengths),
                  ('Ground truth answers per query', correct_answers),
                  ('Candidates per query', candidates_count)]

    for label, data in statistics:
        print()
        print(label)
        print_stats(data)
        print('-' * 30)


if __name__ == '__main__':
    main()
