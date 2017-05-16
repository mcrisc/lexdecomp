# LexDecomp

LexDecomp is an implementation of the Answer Selection (AS) model proposed in the paper *Sentence Similarity Learning by Lexical Decomposition and Composition*, by [(Wang et al., 2016)][wang-2016]. It is organized in two main modules:

[wang-2016]: https://arxiv.org/abs/1602.07019


**1. Decomposition**

In this module (`lexdecomp/decomp.py`) are implemented the paper main ideas, as described in sections 3.1 (Semantic Matching Functions) and 3.2 (Decomposition Functions).

The script takes tokenized text as input and generates matrix files as output. Since the method is deterministic, no training is required. Vector computations are done by [NumPy][numpy].


**2. Deep Neural Network**

In this module (`lexdecomp/compmodel.py`) is defined the Deep Neural Network (DNN) described in the paper section 3.3 (Composition Functions). It takes matrices generated in the decomposition phase, representing questions and candidate answers, and predicts relevance scores for each candidate answer.

The neural network is implemented in [TensorFlow][tensorflow].


[tensorflow]: https://www.tensorflow.org/
[numpy]: http://www.numpy.org/



## Repository Organization

The repository is organized as follows:
- `lexdecomp/`:
    source files
- `trec-qa/`:
    an answer selection dataset (please check `trec-qa/README`)


## Third-party Libraries
Specifications of third-party libraries can be found in `requirements.txt`.


## Usage

Here is a step-by-step example of how to train and evaluate the model.

### 1. Dataset preparation

1.1 Converting JSONL files to TSV:

    $ cd trec-qa
    $ python3 tools/jsonl2tsv.py dev-filtered.jsonl
    $ python3 tools/jsonl2tsv.py test-filtered.jsonl
    $ python3 tools/jsonl2tsv.py train-filtered.jsonl


1.2 Tokenizing TSV files:
<a name="tokenizing"></a>

    $ python3 ../lexdecomp/dataprep.py dev-filtered.tsv
    $ python3 ../lexdecomp/dataprep.py test-filtered.tsv
    $ python3 ../lexdecomp/dataprep.py train-filtered.tsv

Tokenized files are written to `.txt` files. You'll need them later, in [decomposition step](#decomposition).


1.3 Generating TREC relevance file (qrel):

    $ python3 tools/tsv2trecqrels.py test-filtered.tsv

The relevance file (`test-filtered.qrel`) will be used later to evaluate the trained model.


### 2. Semantic Matching and Decomposition

#### Word Vectors
Word embeddings will be required to run the decomposition. In this example, word2vec vectors pre-trained in Google News were used (available [here](https://code.google.com/archive/p/word2vec/)).

The implementation requires vectors to be stored in two files: `<base-name>.voc` and `<base-name>.npy`, which are, respectively, the vocabulary and vector files. The vocabulary (`.voc`) should be a plain text file containing one word per line, while the vector file (`.npy`) should contain a NumPy matrix, where each line is the vector of the corresponding word in the vocabulary.

You can use the scripts in `tools` directory to get both the NumPy matrix and the vocabulary file from the original word2vec file (`.bin`).



#### Decomposition
<a name="decomposition"></a>
Semantic matching and decomposition are executed as follows:

    $ python3 lexdecomp/decomp.py GoogleNews-vectors-300d.npy trec-qa/dev-filtered.txt dev-filtered.hdf5
    $ python3 lexdecomp/decomp.py GoogleNews-vectors-300d.npy trec-qa/test-filtered.txt test-filtered.hdf5
    $ python3 lexdecomp/decomp.py GoogleNews-vectors-300d.npy trec-qa/train-filtered.txt train-filtered.hdf5

Important notes:
- These commands should be run from the repository base directory.
- The input files *are not* the TSV files generated in step 1.1, but the tokenized files from [step 1.2](#tokenizing).

### 3. Model Training

To train the model:

    $ python3 lexdecomp/train.py {train,dev,test}-filtered.hdf5 saved-model


### 4. Model Evaluation
The script `train.py` prints some measures during training. Although they are perfectly useful to guide the training process, their values differ significantly from those computed by [trec_eval][trec-eval], the "official" measurement tool for the Answer Selection (AS) task.

[trec-eval]: http://trec.nist.gov/trec_eval/

To compute the measures using trec_eval, [download the tool][trec-eval] and execute the following command:

    $ trec_eval.8.1/trec_eval trec-qa/test-filtered.qrels saved-model/test_best-model.results | head


---
