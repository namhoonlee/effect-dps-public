import os
import sys
import argparse
import tensorflow as tf

from dataset import Dataset
from model import Model
import prune
import train
import test
import batch_science
import metaparameters
import check


def parse_arguments():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--nruns', type=int, default=100, help='number of times to run the program')
    # Data
    parser.add_argument('--path_data', type=str, default='path_to_dataset', help='location of data sets')
    parser.add_argument('--datasource', type=str, default='mnist', help='data set to use')
    # Model
    parser.add_argument('--arch', type=str, default='simple-cnn', help='model architecture')
    parser.add_argument('--target_sparsity', type=float, default=0.9, help='target sparsity')
    # Train
    parser.add_argument('--batch_size', type=int, default=100, help='number of examples in mini-batch')
    parser.add_argument('--train_iterations', type=int, default=10000, help='number of training iterations')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer of choice')
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='initial learning rate')
    parser.add_argument('--decay_type', type=str, default='constant', help='learning rate decay type')
    parser.add_argument('--decay_steps', type=int, default=1000, help='period to decay a learning rate')
    parser.add_argument('--end_learning_rate_factor', type=float, default=0.01, help='end learning rate decay factor')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum value')
    # Log
    parser.add_argument('--no_load_cache', action='store_true', help='do not allow loading cache in test mode')
    parser.add_argument('--check_interval', type=int, default=100, help='check interval during training')
    parser.add_argument('--save_interval', type=int, default=1000, help='save interval during training')
    # Batch science
    parser.add_argument('--mparams_kind', nargs='+', type=str, default=[], help='metaparameters to search.')
    parser.add_argument('--check_interval_arrival', type=int, default=100)
    parser.add_argument('--goal_error', type=float, default=0.01, help='a goal error to reach.')
    parser.add_argument('--check_lipschitz', action='store_true', help='check lipschitz constant')
    parser.add_argument('--lipschitz_interval', type=int, default=1000)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    # Set up for project
    study = batch_science.setup_study(args)
    mparams = metaparameters.load(args, study)

    for run in range(args.nruns):
        print('-- Start run ({})'.format(run))

        # Set paths
        path_save = 'run-{}'.format(run)
        path_keys = ['model', 'log', 'assess', 'batch-science', 'lipschitz']
        args.path = {key: os.path.join(path_save, key) for key in path_keys}

        # Update metaparameters for the current run
        args = metaparameters.update(args, mparams, run)

        # Reset the default graph and set a graph-level seed
        tf.reset_default_graph()
        tf.set_random_seed(seed=run)

        # Dataset
        dataset = Dataset(**vars(args))

        # Model
        model = Model(**vars(args))
        model.construct_model()

        # Session
        sess = tf.InteractiveSession()

        # Initialization
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        # Prune
        prune.prune(args, model, sess, dataset)

        # Check Lipschitz smoothness
        if args.check_lipschitz:
            check.lipschitz_smoothness(args, model, sess, dataset)

        # Train and test
        train.train(args, model, sess, dataset)
        test.test(args, model, sess, dataset)

        # Closing
        sess.close()
        print('-- Finish run ({})'.format(run))
        with open(os.path.join(path_save, 'run-finished'), 'w') as fp:
            pass

    sys.exit()


if __name__ == "__main__":
    main()
