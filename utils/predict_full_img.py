from itertools import product
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from PIL import Image
from scipy.misc import toimage


def predict_full_image(img_batch, model, nmb_classes=2):
    """
    Create a segmentation prediction for the whole img,
    concatenating predictions for smaller images of crop_size
    :param img_batch: batch of images of full size
    :param model: nn model, that should have model.predict method,
    size of input image model.input_img_shape and
    size of output image model.pred_img_shape, tuples or list of size 2
    :param nmb_classes: int, number of classes, of size of second dimension in prediction
    :return: a batch of predicted images of shape
        [img_batch[0], 1, target_crop_size[:]*nmb_steps[:],
        where nmb_steps is number of smaller images in the full one
    """
    crop_size = model.input_img_shape
    target_crop_size = model.pred_img_shape

    nmb_imgs, nmb_channels, h, w = img_batch.shape
    edge_size = [(crop_size[i] - target_crop_size[i]) // 2 for i in range(2)]

    step_y_max = (h - 2 * edge_size[0]) // target_crop_size[0]
    step_x_max = (w - 2 * edge_size[1]) // target_crop_size[1]
    nmb_preds = step_y_max*step_x_max

    h_pred, w_pred = target_crop_size[0]*step_y_max, target_crop_size[1]*step_x_max
    pred = np.zeros((nmb_imgs, nmb_classes, h_pred, w_pred))

    print 'Overall:{}'.format(nmb_preds),
    for i, (step_y, step_x) in enumerate(product(np.arange(step_y_max), np.arange(step_x_max))):

        if i % 10 == 0:
            print i,

        y = step_y * target_crop_size[0]
        x = step_x * target_crop_size[0]

        data = img_batch[:, :, y:y + crop_size[0], x:x + crop_size[1]]
        # shape [batch_size, nmb_classes=2 (or check parameters of model),
        # output_shape[0], output_shape[1]]
        res = model.predict(data)
        if not (1.0 - 1e-4 < res[0,:,0,0].sum() < 1.0 + 1e-4):
            print res[0,:,0,0].sum()
        assert (1.0 - 1e-4 < res[0,:,0,0].sum() < 1.0 + 1e-4)
        # WARN: CHANGE SLICE OF RES IF WANT, but change NMB_CLASSES ALSO
        pred[:, :, y:y+target_crop_size[0], x:x+target_crop_size[1]] = np.copy(res[:,:,:,:])
    return pred


def process_img_for_plot(img, gt_segm, pred, rate=0.2, masked_img=None):

    edge_img_pred = [(img.shape[i] - pred.shape[i]) // 2 for i in range(2)]
    edge_img_gt = [(img.shape[i] - gt_segm.shape[i]) // 2 for i in range(2)]
    edge_gt_pred = [(gt_segm.shape[i] - pred.shape[i]) // 2 for i in range(2)]

    # working part of img or masked_img (if masked_img is not None)
    frame = np.ones(img.shape[:2])
    frame[edge_img_pred[0]:-edge_img_pred[0], edge_img_pred[1]:-edge_img_pred[1]] = 0.
    if masked_img is None:
        framed_img = np.array(overlayFrame(toimage(img), toimage(frame), rate=rate))
    else:
        framed_img = np.array(overlayFrame(toimage(masked_img), toimage(frame), rate=rate))

    # img vs pred
    if edge_img_pred[0] != 0:
        img_cropped = img[edge_img_pred[0]:-edge_img_pred[0], edge_img_pred[1]:-edge_img_pred[1], :]
    else:
        img_cropped = img
    img_vs_pred = overlay(img_cropped, pred, rate=rate)

    # img vs gt_segm
    # fit img to pred shape
    if edge_img_pred[0] != 0:
        img_gt_cropped = img[edge_img_pred[0]:-edge_img_pred[0], edge_img_pred[1]:-edge_img_pred[1], :]
    else:
        img_gt_cropped = img
    # fit gt to pred shape
    if edge_gt_pred[0] != 0:
        gt_cropped = gt_segm[edge_gt_pred[0]:-edge_gt_pred[0], edge_gt_pred[1]:-edge_gt_pred[1]]
    else:
        gt_cropped = gt_segm
    img_vs_gt = overlay(img_gt_cropped, gt_cropped, rate=rate)

    # diff gt - prediction
    if edge_gt_pred[0] != 0:
        gt_cropped = gt_segm[edge_gt_pred[0]:-edge_gt_pred[0], edge_gt_pred[1]:-edge_gt_pred[1]]
    else:
        gt_cropped = gt_segm

    diff = gt_cropped - pred
    # IF PRED IS TOO SMALL! ->
    #diff = gt_cropped - pred*1./pred.max()
    return framed_img,  gt_cropped, img_vs_gt, img_vs_pred, diff


def overlayFrame(backImage, overImage, rate=0.5):
    background = backImage.convert("RGBA")
    overlay = overImage.convert('RGBA')

    return Image.blend(background, overlay, rate)


def overlay(rgb_image, pred_matrix, rate=0.5, color_channel=0):
    
    pred_expand = np.expand_dims(255*pred_matrix, axis=2)
    wsum = rate*pred_expand + (1-rate)*rgb_image[:,:,color_channel:color_channel+1]
    new_img = np.copy(rgb_image)
    new_img[:,:,color_channel:color_channel+1] = wsum
    
    return new_img


def plot_dsp(pred, img, gt_segm, img_gt, img_pred, gt_pred, pred_map2 = None, bin_pred=None, path_to_save=None, show=False):
    """
    Plot input image, predicted image and ground truth segmentation
    :param gt_pred:
    :param img_pred:
    :param img_gt:
    :param img: image
    :param gt_segm: image
    :param pred: image
    :param path_to_save: path to save image
    :param show: bool, whether to show plot or not
    :return:
    """
    fig = plt.figure(figsize=(24, 12))

    ax1 = fig.add_subplot(231)
    ax1.imshow(img)
    ax1.set_title('input')

    ax2 = fig.add_subplot(232)
    ax2.imshow(gt_segm, cmap=cm.coolwarm, vmin=0, vmax=1)
    ax2.set_title('ground truth')

    ax3 = fig.add_subplot(233)
    im3=ax3.imshow(pred, cmap=cm.coolwarm, vmin=0, vmax=1)
    ax3.set_title('pred map lesion.\nmax = %.3f'%pred.max())
    fig.colorbar(im3, ax=ax3)

    ax4 = fig.add_subplot(234)
    im4 = ax4.imshow(gt_pred, cmap=cm.spectral, vmin=-1, vmax=1)
    ax4.set_title('diff: gt - pred')
    fig.colorbar(im4, ax=ax4)

    ax5 = fig.add_subplot(235)
    if bin_pred is None:
        im5 = ax5.imshow(abs(gt_pred), cmap=cm.coolwarm, vmin=0, vmax=1)
        ax5.set_title('abs_diff: |gt - pred|')
        fig.colorbar(im5, ax=ax5)
    else:
        ax5.imshow(bin_pred, cmap=cm.coolwarm, )
        ax5.set_title('bin_pred: argmax.\nmax = %.3f' % bin_pred.max())

    ax6 = fig.add_subplot(236)
    if pred_map2 is None:
        ax6.imshow(img_gt)
        ax6.set_title('input with gt mask')
    else:
        im6=ax6.imshow(pred_map2, cmap=cm.coolwarm, vmin=0, vmax=1)
        ax6.set_title('pred map nonlesion.\nmax = %.3f' % pred_map2.max())
        fig.colorbar(im6, ax=ax6)

    if show:
        plt.show()

    if path_to_save:
        plt.savefig(path_to_save)
    plt.close()
    
    