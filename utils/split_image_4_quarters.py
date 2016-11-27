""" Function related to images preprocessing """

import os
import cv2
import itertools


def split_img_quarters(img_path, dir_to_save):
    """
    Splits image located in img_path into to 4 quarter images,
      Saves to dir_to_save with names img_path_i, where i in [0,1,2,3]
    :param img_path: path to image to split
    :param dir_to_save: path to directory to save quarter images
    :return:
    """

    basename = os.path.basename(img_path).split('.')[0]
    img = cv2.imread(img_path)
    (imgheight, imgwidth) = img.shape[:2]

    height = imgheight / 2
    width = imgwidth / 2

    img_quarters = []
    for i, (h, w) in enumerate(itertools.product([0, 1], repeat=2)):
        img_quarters.append(img[h*height: (h+1)*height, w*width: (w+1)*width])
        path = os.path.join(dir_to_save, basename+'_{}.JPG'.format(i))
        cv2.imwrite(path, img_quarters[i])
