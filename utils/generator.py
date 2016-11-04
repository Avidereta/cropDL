import numpy as np
import Queue
import threading
import matplotlib.pyplot as plt


def batch_generator(data, target, batch_size, random_seed=None):
    """
    Having np.array of images and corresponding target on the each image
    generates sets of data (data_batch, target_batch) of size batch_size.

    :param data: np.array of images
    :param: target: np.array of binary images
    :param batch_size: int, sizeof batch
    :param random_seed: random seed
    :return: data_batch: data sequence of batch_size
             target_batch: data sequence of batch_size
    """
    np.random.seed(seed=random_seed)
    while True:
        idxs = np.random.randint(0, data.shape[0], batch_size)
        yield data[idxs], target[idxs]


def batch_generator_from_paths(data_paths, target_paths, batch_size, random_seed=None):
    """
    Having np.array of images and corresponding target on the each image
    generates sets of data (data_batch, target_batch) of size batch_size.

    :param data_paths: list of paths to images
    :param target_paths: list of path to corresponded targets
    :param batch_size: int, sizeof batch
    :param random_seed: random seed
    :return: data_batch: data sequence of batch_size
             target_batch: data sequence of batch_size
    """
    np.random.seed(seed=random_seed)
    if len(data_paths) != len(target_paths):
        raise ValueError("Inconsistent lists of data paths and targets paths")

    img_data_size = plt.imread(data_paths[0])
    img_target_size = plt.imread(target_paths[0])

    while True:
        data = np.zeros((batch_size, 3, img_data_size[0], img_data_size[1]), dtype=np.uint8)
        target = np.zeros((batch_size, 1, img_target_size[0], img_target_size[1]), dtype=np.uint8)
        idxs = np.random.randint(0, len(data_paths), batch_size)
        for i, idx in enumerate(idxs):
            im, gt_im = data_paths[idx], target_paths[idx]
            data[i] = plt.imread(im).transpose((2, 0, 1))
            target[i, 0] = plt.imread(gt_im, 0) / 256.
        yield data, target


def iterate_over(generator, n_batches):
    for i in range(n_batches):
        yield generator.next()


def random_crop_generator(generator, crop_size=(128, 128), target_crop_size=None,
                          info=False, bin_threshold=0.5, info_threshold=0.05):
    """
    Yields a random crop of batch produces by generator
    :param generator: batch generator which yields batch of pairs data and target
    :param crop_size: tuple, list of int or int with desired size of data
    :param target_crop_size: tuple, list of int or int with desired size of target
    None if the desired size for target and crop_size for data are the same
    :param info: bool, whether to generate or not batches of with certain percentage of
    elements larger than bin_threshold
    :param bin_threshold: float, if info is True: threshold for dividing elements into 2 groups
    :param info_threshold: float, if info is True: percentage of elements larger than bin_threshold
    :return: yields batch of data and target pairs
    """
    np.random.seed()
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size, crop_size]
    elif len(crop_size) == 2:
        crop_size = list(crop_size)
    else:
        raise ValueError("invalid crop_size")
    if target_crop_size is None:
        target_crop_size = crop_size
    else:
        if type(target_crop_size) not in (tuple, list):
            target_crop_size = [target_crop_size, target_crop_size]
        elif len(target_crop_size) == 2:
            target_crop_size = list(target_crop_size)
        else:
            raise ValueError("invalid target_crop_size")

    for data, target in generator:
        lb_x = np.random.randint(0, data.shape[2] - crop_size[0])
        lb_y = np.random.randint(0, data.shape[3] - crop_size[1])
        data = data[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1]]
        target = target[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1]]

        if target_crop_size != crop_size:
            width, height = crop_size[0], crop_size[1]
            new_width, new_height = target_crop_size[0], target_crop_size[1]
            ## TODO: debus type checker
            if new_width or new_height not in (integer):
                raise ValueError('invalid type for target_crop_size elements')
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2
            target = target[:, :, left:right, top:bottom]
        if info:
            info_percent = np.sum(bin_threshold < target) * 1. / np.size(target.ravel())
            if info_percent > info_threshold:
                yield data, target
            else:
                pass
        else:
            yield data, target


def threaded_generator(generator, num_cached=10):
    """
    generates batches in advance
    :param generator: generator to use
    :param num_cached: int, number of batches to cache from one iteration
    :return:
    """

    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()