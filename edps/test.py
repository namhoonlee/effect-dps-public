import os
import tensorflow as tf
import glob

import evaluate
from helpers import cache_json


def test(args, model, sess, dataset):
    print('-- Test')

    saver = tf.train.Saver()

    # Identify which checkpoints are available.
    all_model_checkpoint_paths = glob.glob(os.path.join(args.path['model'], 'itr-*'))
    all_model_checkpoint_paths = list(set([f.split('.')[0] for f in all_model_checkpoint_paths]))
    model_files = {int(s[s.index('itr')+4:]): s for s in  all_model_checkpoint_paths}

    # Subset of iterations.
    itrs = sorted(model_files.keys())
    itr_subset = itrs
    assert itr_subset

    # Evaluate.
    acc = []
    for itr in itr_subset:
        print('evaluation: {} | itr-{}'.format(dataset.datasource, itr))
        # run evaluate and/or cache
        result = cache_json(
            os.path.join(args.path['assess'], 'itr-{}.json'.format(itr)),
            lambda: evaluate.eval(model, sess, dataset, 'test', saver, model_files[itr]),
            makedir=True,
            allow_load=(not args.no_load_cache))
        # print
        print('Accuracy: {:.5f} (#examples:{})'.format(result['accuracy'], result['num_example']))
        acc.append(result['accuracy'])
    print('Max: {:.5f}, Min: {:.5f} (#Eval: {})'.format(max(acc), min(acc), len(acc)))
    pstr = 'Error: {:.3f} %'.format((1 - max(acc))*100)
