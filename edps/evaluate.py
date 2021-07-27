import numpy as np


def eval(model, sess, dataset, split, saver=None, model_file=None):
    # load model
    if saver is not None and model_file is not None:
        try:
            saver.restore(sess, model_file)
        except:
            raise FileNotFoundError

    # load test set
    generator = dataset.get_generator('epoch', split, False)

    # run
    accuracy = []
    while True:
        batch = dataset.get_next_batch(100, generator)
        if batch is not None:
            feed_dict = {}
            feed_dict.update({model.inputs[key]: batch[key] for key in ['image', 'label']})
            feed_dict.update({model.compress: False, model.is_train: False})
            result = sess.run([model.outputs], feed_dict)
            accuracy.extend(result[0]['acc_individual'])
        else:
            break

    results = { # has to be JSON serialiazable
        'accuracy': np.mean(accuracy).astype(float),
        'num_example': len(accuracy),
    }
    assert results['num_example'] == dataset.num_example[split]
    return results
