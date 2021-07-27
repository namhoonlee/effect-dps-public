import os
import tensorflow as tf
import time
import numpy as np
import pickle
import math

import batch_science
import evaluate
import check


def train(args, model, sess, dataset):
    print('-- Train')
    t_start = time.time()

    for key in ['model', 'log']:
        if not os.path.isdir(args.path[key]):
            os.makedirs(args.path[key])
    saver = tf.train.Saver()

    logs = {
        'train': {'itr': [], 'los': [], 'acc': []},
        'val': {'itr': [], 'los': [], 'acc': []},
        'lipschitz': {'itr': [], 'val': []},
    }

    generators = {}
    generators['train'] = dataset.get_generator('unlimited', 'train', True)
    generators['val'] = dataset.get_generator('unlimited', 'val', True)

    bs_status = 'incomplete'
    for itr in range(args.train_iterations):
        batch = dataset.get_next_batch(args.batch_size, generators['train'])
        feed_dict = {}
        feed_dict.update({model.inputs[key]: batch[key] for key in ['image', 'label']})
        feed_dict.update({model.is_train: True})
        input_tensors = [model.outputs]
        if (itr+1) % args.check_interval == 0:
            input_tensors.extend([model.sparsity])
        input_tensors.extend([model.train_op])
        result = sess.run(input_tensors, feed_dict)
        logs['train'] = _update_logs(logs['train'],
            {'itr': itr+1, 'los': result[0]['los'], 'acc': result[0]['acc']})

        # Check on validation set
        if (itr+1) % args.check_interval == 0:
            batch = dataset.get_next_batch(args.batch_size, generators['val'])
            feed_dict = {}
            feed_dict.update({model.inputs[key]: batch[key] for key in ['image', 'label']})
            input_tensors = [model.outputs]
            result_val = sess.run(input_tensors, feed_dict)
            logs['val'] = _update_logs(logs['val'],
                {'itr': itr+1, 'los': result_val[0]['los'], 'acc': result_val[0]['acc']})

        # Print results
        if (itr+1) % args.check_interval == 0:
            pstr = '(train/val) los:{:.3f}/{:.3f} acc:{:.3f}/{:.3f} spa:{:.3f}'.format(
                result[0]['los'], result_val[0]['los'],
                result[0]['acc'], result_val[0]['acc'],
                result[1],
            )
            print('itr{}: {} (t:{:.1f})'.format(itr+1, pstr, time.time() - t_start))
            t_start = time.time()

        # Save
        if (itr+1) % args.save_interval == 0:
            saver.save(sess, args.path['model'] + '/itr-' + str(itr))
            _save_logs(os.path.join(args.path['log'], 'train.pickle'), logs['train'])
            _save_logs(os.path.join(args.path['log'], 'val.pickle'), logs['val'])

        # Check if the model hit the goal error
        if (itr+1) % args.check_interval_arrival == 0:
            results = evaluate.eval(model, sess, dataset, 'val')
            err = 1 - results['accuracy']
            pstr = '(entire val) acc:{:.3f} err:{:.3f}'.format(results['accuracy'], err)
            print('itr{}: {} (t:{:.1f})'.format(itr+1, pstr, time.time() - t_start))
            t_start = time.time()
            if err <= args.goal_error:
                bs_status = 'complete'
                steps_to_result = itr+1
                print('-- batch science')
                print('Hit the goal (steps to result: {})'.format(steps_to_result))
                batch_science.save_results(args, steps_to_result, status=bs_status)
                break
            elif math.isnan(err) or math.isnan(result[0]['los']):
                bs_status = 'infeasible'
                print('-- batch science')
                print('Training infeasible')
                batch_science.save_results(args, steps_to_result=-2, status=bs_status)
                break
            else:
                pass

        # Lipschitz smoothness.
        if args.check_lipschitz:
            lipschitz = None
            if (itr+1) % args.lipschitz_interval == args.lipschitz_interval - 1:
                grad_prev_vec = check.lipschitz_smoothness(args, model, sess, dataset)
            elif (itr+1) % args.lipschitz_interval == 0:
                lipschitz = check.lipschitz_smoothness(args, model, sess, dataset, grad_prev_vec)
            else:
                pass
            if lipschitz is not None:
                print('itr{}: lipschitz smoothness: {:.3f} (t:{:.1f})'.format(
                    itr+1, lipschitz, time.time() - t_start))
                t_start = time.time()
                logs['lipschitz'] = _update_logs(logs['lipschitz'],
                    {'itr': itr+1, 'val': lipschitz})
                _save_logs(os.path.join(args.path['log'], 'lipschitz.pickle'), logs['lipschitz'])

    if bs_status not in ['complete', 'infeasible']:
        print('-- batch science')
        print('Never reached to the goal.')
        batch_science.save_results(args, steps_to_result=-1, status=bs_status)


def _update_logs(logs, log):
    for key in logs.keys():
        logs[key].extend([log[key]])
    return logs

def _save_logs(filename, results):
    with open(filename, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
