import numpy as np


def lipschitz_smoothness(args, model, sess, dataset, grad_prev_vec=None):
    # For convenience.
    split = 'train'
    num_example = dataset.num_example[split]
    bs = 1000
    nitr = num_example // bs
    remainder = num_example % bs
    feed_keys = ['image', 'label']
    def _vectorize_list_narrays(list_narrays):
        list_1d = [np.reshape(narray, -1) for narray in list_narrays]
        vector = np.concatenate(list_1d, axis=0)
        return vector

    # Compute Lipschitz smoothness.
    if grad_prev_vec is None: # preparation step.
        generator = dataset.get_generator('epoch', split, False)

        # Store the current weights as previous weights.
        feed_dict = {}
        feed_dict.update({model.update_w_prev: True})
        input_tensors = [model.w_prev] # w_prev must be executed.
        _ = sess.run(input_tensors, feed_dict)

        # Calculate gradients w.r.t. the current weights.
        grad_prev_vec = []
        for i in range(nitr):
            feed_dict = {}
            batch = dataset.get_next_batch(bs, generator)
            feed_dict.update({model.inputs[key]: batch[key] for key in feed_keys})
            input_tensors = [model.grad]
            result_lipschitz = sess.run(input_tensors, feed_dict)
            grad_prev_vec.append(_vectorize_list_narrays(result_lipschitz[0]) * bs)
        if remainder > 0:
            feed_dict = {}
            batch = dataset.get_next_batch(remainder, generator)
            feed_dict.update({model.inputs[key]: batch[key] for key in feed_keys})
            input_tensors = [model.grad]
            result_lipschitz = sess.run(input_tensors, feed_dict)
            grad_prev_vec.append(_vectorize_list_narrays(result_lipschitz[0]) * remainder)
        grad_prev_vec = np.sum(grad_prev_vec, axis=0) / num_example

        return grad_prev_vec
    else: # measure step.

        # Copy the current weights.
        feed_dict = {}
        feed_dict.update({model.update_w_copy: True})
        input_tensors = [model.w_copy] # w_copy must be executed.
        _ = sess.run(input_tensors, feed_dict)

        # Calculate gradients w.r.t. the new weights.
        lipschitz_all = []

        num_gamma = 10
        gammas = np.linspace(start=0.0, stop=1.0, num=num_gamma, endpoint=False)
        gammas = gammas[1:] # omit the first gamma (0.0).

        for gamma in gammas:
            generator = dataset.get_generator('epoch', split, False)

            grad_new_vec = []
            for i in range(nitr):
                feed_dict = {}
                feed_dict.update({model.update_weights: True})
                feed_dict.update({model.gamma: gamma})
                input_tensors = [model.w_new]
                _ = sess.run(input_tensors, feed_dict)

                feed_dict = {}
                batch = dataset.get_next_batch(bs, generator)
                feed_dict.update({model.inputs[key]: batch[key] for key in feed_keys})
                feed_dict.update({model.gamma: gamma})
                input_tensors = [model.grad]
                result_lipschitz = sess.run(input_tensors, feed_dict)
                grad_new_vec.append(_vectorize_list_narrays(result_lipschitz[0]) * bs)

                feed_dict = {}
                feed_dict.update({model.gamma: gamma})
                input_tensors = [model.denom]
                result_lipschitz = sess.run(input_tensors, feed_dict)
                denom = result_lipschitz[0]

            if remainder > 0:
                feed_dict = {}
                feed_dict.update({model.update_weights: True})
                feed_dict.update({model.gamma: gamma})
                input_tensors = [model.w_new]
                _ = sess.run(input_tensors, feed_dict)

                feed_dict = {}
                batch = dataset.get_next_batch(remainder, generator)
                feed_dict.update({model.inputs[key]: batch[key] for key in feed_keys})
                feed_dict.update({model.gamma: gamma})
                input_tensors = [model.grad]
                result_lipschitz = sess.run(input_tensors, feed_dict)
                grad_new_vec.append(_vectorize_list_narrays(result_lipschitz[0]) * remainder)

                feed_dict = {}
                feed_dict.update({model.gamma: gamma})
                input_tensors = [model.denom]
                result_lipschitz = sess.run(input_tensors, feed_dict)
                denom = result_lipschitz[0]

            grad_new_vec = np.sum(grad_new_vec, axis=0) / num_example

            grad_diff = grad_new_vec - grad_prev_vec
            lipschitz_gamma = np.linalg.norm(grad_diff) / denom
            lipschitz_all.append(lipschitz_gamma)

        lipschitz = np.max(lipschitz_all)
        return lipschitz
