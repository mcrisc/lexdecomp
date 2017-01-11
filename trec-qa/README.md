# TREC-QA Answer Selection Dataset

This the Answer Sentence Selection dataset (QASent), provided by [Mengqiu Wang](http://cs.stanford.edu/people/mengqiu/) and first used in [Wang et al. (2007)](http://www.aclweb.org/anthology/D/D07/D07-1003.pdf). Since then, it became the [standard benchmark][acl-wiki] for the Answer Selection task. This dataset is  formed by data collected from TREC-QA track.

[acl-wiki]: http://www.aclweb.org/aclwiki/index.php?title=Question_Answering_(State_of_the_art)


## Dataset Partition

The dataset is partitioned into 3 subsets: TRAIN, DEV and TEST. There's also an extra subset, TRAIN-ALL (automatically annotated), that contains a larger number of training examples.

Subset | # of Questions | Annotation
---- | -------------- | ---------
TRAIN | 94 | manual
DEV | 82 | manual
TEST | 100 | manual
TRAIN-ALL | 1,229 | automatic


### Distributed Files

The dataset is distributed as [JSONL files](http://jsonlines.org/) named after the corresponding partition (`{train,dev,test,train-all}.jsonl`). Each file contains a set of questions and corresponding labeled candidate answers.

Please check the file contents for record format. It's straightforward.


### Filtered Version

Some of the original questions have only one positive (correct) answer, or have only negative answers. There's also a filtered version of the dataset (`*-filtered.jsonl`) with these questions removed. Filtered partitions contains only questions with at least one positive and one negative answer.

Filtered Subset | # of Questions
---- | --------------
TRAIN | 89
DEV | 69
TEST | 68
TRAIN-ALL | 1,161


### Original Files

This distribution was organized from files found in [this repository][jacana] (also [forked here][jacana-fork]).


[jacana]: https://github.com/xuchen/jacana/tree/master/tree-edit-data/answerSelectionExperiments/data
[jacana-fork]: https://github.com/mcrisc/jacana/tree/master/tree-edit-data/answerSelectionExperiments/data


The table below relates distributed files to their original versions.

Distributed | Original
----------- | --------
train.jsonl | train.xml
dev.jsonl | dev.xml
test.jsonl | test-less-than-40.xml
train-all.jsonl | train2393.xml.gz


Conversions were done automatically by scripts found in directory `tools`.

---
