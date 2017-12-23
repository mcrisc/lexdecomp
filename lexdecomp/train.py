"""Trains the sentence composition model.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score as average_precision

import compmodel
import dataio
from metrics import mean_reciprocal_rank


EMBEDDING_SIZE = 300
EPOCHS = 3
KEEP_PROB = 0.5
STOP_AFTER = 5  # STOP_AFTER = 0 disables early stopping


def compute_metrics(sess, logits_op, placeholders, data_file, exporter=None):
    """Compute metrics MAP and MRR over a dataset.

    :param sess: TensorFlow session
    :param logits_op: an operation that returns the scores for a given set of
    sentences
    :param placeholders: placeholders defined for `logits_op`
    :data_file: a HDF5 file object holding the dataset

    :returns: the values of MAP and MRR as a tuple: (MAP, MRR)
    """
    questions_ph, sentences_ph, keep_prob_ph = placeholders

    if exporter is None:
        exporter = dataio.no_op()
    next(exporter)  # priming the coroutine

    total_avep = 0.0
    total_mrr = 0.0
    n_questions = 0
    for batch in dataio.question_batches(data_file):
        feed_dict = {
            questions_ph: batch.questions,
            sentences_ph: batch.sentences,
            keep_prob_ph: 1.0
        }
        scores = logits_op.eval(session=sess, feed_dict=feed_dict)
        exporter.send(scores)

        n_questions += 1
        avep = average_precision(batch.labels, scores)
        total_avep += avep
        mrr = mean_reciprocal_rank(batch.labels, scores)
        total_mrr += mrr
    exporter.close()

    mean_avep = total_avep / n_questions
    mean_mrr = total_mrr / n_questions
    return mean_avep, mean_mrr


def run_training(training_data, dev_data, test_data, model_dir):
    # reading metadata
    max_question = np.asscalar(
        training_data['metadata/questions/max-size'][...])
    max_sentence = np.asscalar(
        training_data['metadata/sentences/max-size'][...])
    IN_CHANNELS = 2

    print('building the graph')
    # placeholders
    questions = tf.placeholder(
        tf.float32,
        [None, EMBEDDING_SIZE, max_question, IN_CHANNELS])
    sentences = tf.placeholder(
        tf.float32,
        [None, EMBEDDING_SIZE, max_sentence, IN_CHANNELS])
    labels = tf.placeholder(tf.float32, [None])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    # building the graph
    print('HERE!-0')
    logits = compmodel.inference(questions, sentences, keep_prob)
    print('HERE!-1')
    loss = compmodel.loss(logits, labels)
    print('HERE!-2')
    train_op = compmodel.training(loss)
    print('HERE!')
    saver = tf.train.Saver()
    bestdev_model_file = Path(model_dir, 'best-dev_model.ckpt').as_posix()

    def evaluate(session, dataset, data_label, model_label):
        print()
        print('computing MAP on %s set (%s model)...'
              % (data_label, model_label))
        placeholders = (questions, sentences, keep_prob)
        trec_file = Path(
            model_dir,
            '%s_%s-model.results' % (data_label, model_label)).as_posix()
        exporter = dataio.trec_exporter(trec_file)
        map_test, mrr_test = compute_metrics(
            session, logits, placeholders, dataset, exporter=exporter)
        print('# MAP({0:s}): {1:.3f}, MRR({0:s}): {2:.3f}'.format(
                data_label, map_test, mrr_test))
        print('TREC results exported')

    # training
    print('training: %d epochs' % EPOCHS)
    init = tf.global_variables_initializer()
    best_map = 0.0
    last_improvement = 0

    with tf.Session() as sess:
        sess.run(init)
        step = 0
        for batch in dataio.question_epochs(training_data, EPOCHS):
            step += 1
            print('step:', step)

            feed_train = {
                questions: batch.questions,
                sentences: batch.sentences,
                labels: batch.labels,
                keep_prob: KEEP_PROB}

            _, loss_value = sess.run([train_op, loss], feed_dict=feed_train)

            if step % 10 == 0:
                placeholders = (questions, sentences, keep_prob)
                map_dev, mrr_dev = compute_metrics(
                    sess, logits, placeholders, dev_data)
                print('# Step %d, MAP(dev): %.3f, MRR(dev): %.3f'
                      % (step, map_dev, mrr_dev))
                last_improvement += 1
                if map_dev > best_map:
                    improvement = map_dev - best_map
                    best_map = map_dev
                    last_improvement = 0
                    saver.save(sess, bestdev_model_file)
                    print('Model saved (improved: %.3f)' % improvement)

            if STOP_AFTER > 0 and last_improvement >= STOP_AFTER:
                print('Early stopping...')
                break

        # evaluating models
        print('Saving latest model...')
        latest_model_file = Path(model_dir, 'latest_model.ckpt').as_posix()
        saver.save(sess, latest_model_file)
        evaluate(sess, test_data, 'test', 'latest')

        print()
        print('Restoring best dev model...')
        saver.restore(sess, bestdev_model_file)
        evaluate(sess, test_data, 'test', 'best')


def main():
#    parser = argparse.ArgumentParser(
#        description='Trains the sentence composition model (CNN).')
#    parser.add_argument('training', help='training set (.hdf5)')
#    parser.add_argument('dev', help='dev set (.hdf5)')
#    parser.add_argument('test', help='test set (.hdf5)')
#    parser.add_argument('model_dir', help='directory to save models')
#    args = parser.parse_args()
#    args.append('../train-filtered.hdf5')
#    args.append('../dev-filtered.hdf5')
#    args.append('../test-filtered.hdf5')
#    args.append('../saved-model')
    # checking model directory
#    print ('Yes1')
    model_dir = Path('../saved-model')
    if not model_dir.exists():
        model_dir.mkdir()

    # data files
#    print ('Yes2')
    training_data = h5py.File('../train-filtered.hdf5')
    dev_data = h5py.File('../dev-filtered.hdf5')
    test_data = h5py.File('../test-filtered.hdf5')

    try:
#        print ('Yes3')
        run_training(training_data, dev_data, test_data,
                     '../saved-model')
    finally:
        training_data.close()
        dev_data.close()
        test_data.close()


if __name__ == '__main__':
    main()
