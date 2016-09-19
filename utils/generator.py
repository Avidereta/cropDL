import numpy as np

def generator(X, Y, batch_size, random_seed=None):
    """
    Having np.array of images and corresponding number of lesions on the each image
    generates sets of data (x_batch, y_batch) of size batch_size.

    :param X: np.array of images
    :param batch_size: int, sizeof batch
    :param random_seed: random seed
    :return: x_batch: data sequence of batch_size
             y_batch: data sequence of batch_size
    """
    while True:
        np.random.seed(seed=random_seed)

        idxs = np.random.randint(0, len(X), batch_size)
        yield X[idxs], Y[idxs]


def iterate_over(generator,n_batches):
    for i in range(n_batches):
        yield generator.next()