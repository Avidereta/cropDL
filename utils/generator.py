"""
Utils for data batch generations
"""

import numpy as np
import Queue
import threading
import matplotlib.pyplot as plt


def batch_generator(data, target, batch_size, random_seed=None, masked_data=None):
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
        if masked_data is not None:
            masked_data_idx = masked_data[idx]
        yield data[idxs], target[idxs], masked_data_idx
        
        
def batch_generator_from_paths(data_paths, target_paths, batch_size, random_seed=None,
                              info=False, bin_threshold=0.5, info_threshold=0.01, masked_data_paths=None):
    """
    Having np.array of images and corresponding target on the each image
    generates sets of data (data_batch, target_batch) of size batch_size.

    :param data_paths: list of paths to images
    :param target_paths: list of paths to corresponded targets
    :param batch_size: int, sizeof batch
    :param random_seed: random seed
    :param masked_data_paths: list of paths to corresponded masked images. 
            It is assumed that masked_images are of the same size as images from data paths. 
            This can be helpful if you want to add some additional info on the images for future
            visualization.
    :return: data_batch: data sequence of batch_size
             target_batch: data sequence of batch_size
             masked_data_batch: data sequence of batch_size or, if masked_data_paths is None, None 
    """
    np.random.seed(seed=random_seed)
    if len(data_paths) != len(target_paths):
        raise ValueError("Inconsistent lists of data paths and targets paths")
    
    if masked_data_paths is not None:
        if len(data_paths) != len(masked_data_paths):
            raise ValueError("Inconsistent lists of data paths and masked_data paths")
        img_masked_data_size = plt.imread(masked_data_paths[0]).shape

    img_data_size = plt.imread(data_paths[0]).shape
    img_target_size = plt.imread(target_paths[0]).shape

    while True:
        data = np.zeros((batch_size, 3, img_data_size[0], img_data_size[1]), dtype=np.uint8)
        target = np.zeros((batch_size, 1, img_target_size[0], img_target_size[1]), dtype=np.uint8)
        
        if masked_data_paths is not None:
            masked_data = np.zeros((batch_size, 3, img_masked_data_size[0], img_masked_data_size[1]), dtype=np.uint8)
        else: masked_data = None
            
        idxs = np.random.randint(0, len(data_paths), batch_size)
        for i, idx in enumerate(idxs):
            im, gt_im = data_paths[idx], target_paths[idx]
            data[i] = plt.imread(im).transpose((2, 0, 1))
            target[i, 0] = plt.imread(gt_im, 0) / np.max(plt.imread(gt_im, 0))
            if masked_data_paths is not None:
                masked_im = masked_data_paths[idx]
                masked_data[i] = plt.imread(masked_im).transpose((2,0,1))
        if info:
            info_percent = np.sum(bin_threshold < target) * 1. / np.size(target.ravel())
            if info_percent > info_threshold:
                yield data, target, masked_data
            else:
                pass
        else:
            yield data, target, masked_data

def random_crop_generator(generator, crop_size=(128, 128), target_crop_size=None, random_seed=None,
                          info=False, bin_threshold=0.5, info_threshold=0.005):
    """
    Yields a random crop of batch produces by generator
    :param generator: batch generator which yields batch of data, target and masked_data. 
        Masked data can be None in case no masked data paths are provided in generator arguments. 
    :param crop_size: tuple, list of int or int with desired size of data
    :param target_crop_size: tuple, list of int or int with desired size of target
    None if the desired size for target and crop_size for data are the same
    :param random_seed: random seed
    :param info: bool, whether to generate or not batches of with certain percentage of
    elements larger than bin_threshold
    :param bin_threshold: float, if info is True: threshold for dividing elements into 2 groups
    :param info_threshold: float, if info is True: percentage of elements larger than bin_threshold
    :return: yields batch of data and target pairs
    """
    np.random.seed(seed=random_seed)
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

    for data, target, masked_data in generator:
        lb_x = np.random.randint(0, data.shape[2] - crop_size[0])
        lb_y = np.random.randint(0, data.shape[3] - crop_size[1])
        data = data[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1]]
        target = target[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1]]
        
        if masked_data is not None:
            masked_data = masked_data[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1]]

        if target_crop_size != crop_size:
            width, height = crop_size[0], crop_size[1]
            new_width, new_height = target_crop_size[0], target_crop_size[1]
            # TODO: add integer check
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2
            target = target[:, :, left:right, top:bottom]
        if info:
            info_percent = np.sum(target > bin_threshold) * 1. / np.size(target.ravel())
#             print 'random_crop_generator info_percent %.5f'%info_percent
            if info_percent > info_threshold:
                #print 'random crop generator, info_percent:', info_percent
                yield data, target, masked_data
            else:
                pass
        else:
            yield data, target, masked_data


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


def iterate_over(generator, n_batches):
    for i in range(n_batches):
        yield generator.next()        
            