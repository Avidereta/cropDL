from collections import namedtuple

Sample = namedtuple('Sample', ['ann_path', 'img_path'])


def read_paths(ann_img_path):
    """
    Reads txt file with pairs of paths (e.g annotations and images), creates a list of samples
      with the corresponded pairs.
    :param ann_img_path: txt file with pairs of paths.
    Each line of input file is a pair of paths divided by space
    :return: list of samples
    """
    with open(ann_img_path) as fin:
        samples = [Sample(f.strip().split(' ')[0], f.strip().split(' ')[1]) for f in fin]
    return samples
