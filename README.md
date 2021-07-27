# Understanding the Effects of Data Parallelism and Sparsity on Neural Network Training
This repository contains code for the paper [Understanding the Effects of Data Parallelism and Sparsity on Neural Network Training](https://openreview.net/forum?id=rsogjAnYs4z) (ICLR 2021).

## Prerequisites

### Dependencies
* tensorflow >= 1.14
* python >= 3.6

### Datasets
Put the following datasets in your preferred location (e.g., `./data`).
* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

## Usage
To measure the effect of data parallelism for the workload of Simple-CNN, MNIST and SGD:
```sh
$ python main.py --nruns 100 --path_data PATH_TO_DATA --datasource mnist --arch simple-cnn --target_sparsity 0.9 --batch_size A_BATCH_SIZE --train_iterations 40000 --optimizer sgd --decay_type constant --mparams_kind learning_rate --goal_error 0.02 --check_interval 100
```
To measure the effect of data parallelism for the workload of ResNet-8, CIFAR-10 and Nesterov:
```sh
$ python main.py --nruns 100 --path_data PATH_TO_DATA --datasource cifar-10 --arch resnet-8 --target_sparsity 0.9 --batch_size A_BATCH_SIZE --train_iterations 100000 --optimizer nesterov --decay_type linear --mparams_kind learning_rate momentum decay_steps end_learning_rate_factor --goal_error 0.4 --check_interval 100
```
If you want to measure the Lipschitz constant of gradients, run with `check_lipschitz` flag on.

See `main.py` to run with other options.

## Citation
If you use this code for your work, please cite the following:
```
@inproceedings{lee2021dps,
  title={Understanding the effects of data parallelism and sparsity on neural network training},
  author={Lee, Namhoon and Ajanthan, Thalaiyasingam and Torr, Philip HS and Jaggi, Martin},
  booktitle={ICLR},
  year={2021},
}
```

## License
This project is licensed under the MIT License.
See the [LICENSE](https://github.com/namhoonlee/effect-dps-public/blob/master/LICENSE) file for details.
