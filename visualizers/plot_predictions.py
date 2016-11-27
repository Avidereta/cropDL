from utils.predict_full_img import *


def plot_predictions(pred_fn, generator, path_to_save, ext='png',
                     n_images=10, info=False, info_threshold=0.005, bin_threshold=0.5,
                    pred_nles_les=(0, 1), transp_rate=0.3):
    """
    Plots predictions for generated batch.
    General idea:
        data, target, masked_data = generator()
        pred = pred_fn(data)
        plot(pred)
    :param pred_fn: prediction function
    :param generator: generator which yields batch of data, target and masked data. Take a look at utils.generator
    data - images
    taget - ground truth
    masked_data - images of the same size as data, but with some additional info to visualize. Can be None.
    :param ext: str, image extension 'png', 'JPG'. Default is png.
    :param path_to_save: str, path to save plots.
    Note, that this function generates n_images predictions, thus it will be saved as
    concatenated path_to_save, number and EXT
    :param n_images: int, number of images to save
    :param info: bool, whether to generate or not batches of with certain percentage of
    elements larger than bin_threshold
    :param bin_threshold: float, if info is True: threshold for dividing elements into 2 groups
    :param info_threshold: float, if info is True: percentage of elements larger than bin_threshold
    :param pred_nles_les: tuple of length 2 of ints (int1, int2):
    pred[:,int1,:,:] - non lesion prediction
    pred[:,int2,:,:] - lesion prediction
    :param transp_rate: transparency rate for process_img_for_plot
    """
    cnt_images = 0
    for data, seg, masked_data in generator:
        # shape [batch_size, nmb_classes=2, output_shape. output_shape]
        pred = pred_fn(data)

        # if None transform to list for iteration
        if masked_data is None:
            masked_data = [None]*len(data)

        for i, (d, s, md, p) in enumerate(zip(data, seg, masked_data, pred)):
            # NOTE: CAN BE CHANGED
            p_les = p[pred_nles_les[1]]
            p_nles = p[pred_nles_les[0]]

            # (nmb_channels, h, w) -> (h, w, nmb_channels)
            d_for_plot = d.transpose(1, 2, 0)

            # use only images with informational content (nmb lesion pixels) more than threshold
            if info:
                info_percent = np.sum(s > bin_threshold)*1. / np.size(s.ravel())
                if info_percent < info_threshold:
                    pass
                else:
                    if md is not None:
                        img_edged, gt_edged, plot_img_gt, plot_img_pred, diff_gt_pred = \
                            process_img_for_plot(d_for_plot, s[0], p_les, transp_rate, md.transpose(1, 2, 0))
                    else:
                        img_edged, gt_edged, plot_img_gt, plot_img_pred, diff_gt_pred = \
                            process_img_for_plot(d_for_plot, s[0], p_les, transp_rate)

                    plot_dsp(p_les, img_edged, gt_edged, plot_img_gt, plot_img_pred, diff_gt_pred,
                             pred_map2=p_nles, bin_pred=p.argmax(axis=0),
                             path_to_save=path_to_save+"{}.".format(i)+ext)

                    cnt_images += 1

            # no informational content
            else:
                if md is not None:
                    img_edged, gt_edged, plot_img_gt, plot_img_pred, diff_gt_pred = \
                        process_img_for_plot(d_for_plot, s[0], p_les, transp_rate, md.transpose(1, 2, 0))
                else:
                    img_edged, gt_edged, plot_img_gt, plot_img_pred, diff_gt_pred = \
                        process_img_for_plot(d_for_plot, s[0], p_les, transp_rate)

                plot_dsp(p_les, img_edged, gt_edged, plot_img_gt, plot_img_pred, diff_gt_pred,
                         pred_map2=p_nles, bin_pred=p.argmax(axis=0), path_to_save=path_to_save+"{}.".format(i)+ext)

                cnt_images += 1
        if cnt_images > n_images:
            break
