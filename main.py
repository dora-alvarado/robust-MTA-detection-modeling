import numpy as np
import cv2
from skimage.morphology import skeletonize
from timeit import default_timer as timer

from src.misc_img_func import read_color_img, read_grayscale_img, plot_other
from src.preprocessing import get_gt_points
from src.modeling import hausdorff, eval_splines, draw_gt_vs_model, \
    spline_modeling, mean_closest_distance, draw_mta_model, get_side

n_keypoints = 5
spline_order = 2

np.random.seed(1)
path_img = 'images/images/01_test.tif'
path_model = 'images/output/01_output.jpg'
path_gt = 'images/mta_annotations/01_test.tif'
path_mask = 'images/fov/01_test_mask.gif'
path_unet = 'images/unet/01_test.png'
########################################################################################################################
# Read image
########################################################################################################################
img_color = read_color_img(path_img)
img_mask = read_grayscale_img(path_mask) != 0
img_gt = read_grayscale_img(path_gt)# None
img_unet = read_grayscale_img(path_unet)>0.5# None

if img_gt is not None:
    img_gt = read_grayscale_img(path_gt) > 0.5
    points_gt = get_gt_points(img_gt)

else:
    points_gt = None
since = timer()
enh_img, th_img, dt_im, model, inliers, datapoints, knots = spline_modeling(img_color, img_mask, n_keypoints,
                                                                            spline_order, pixel_error=15.,
                                                                            return_steps=True, th_img=img_unet)

time_elapsed = timer() - since
im_mta = skeletonize(draw_mta_model(th_img, model, keypoints=datapoints, lineThickness=1, truncate = True) / 255.)
im_mta *= img_mask
side = get_side(knots, im_mta.shape[1])

im_comparative = draw_gt_vs_model(img_color*255 , model, points_gt, keypoints=datapoints, side=side, truncate= True)

if side:
    im_mta[:, (im_mta.shape[1] // 2):] = 0
    if img_gt is not None:
        img_gt[:, (im_mta.shape[1] // 2):] = 0
else:
    im_mta[:, :im_mta.shape[1] // 2] = 0
    if img_gt is not None:
        img_gt[:, :im_mta.shape[1] // 2] = 0

if img_gt is not None:
    ske_img_gt_mta = skeletonize(img_gt)
    tp = np.sum((img_gt > 0) & (im_mta > 0))
    p = np.sum(ske_img_gt_mta > 0)
    tpr = tp / p
    tn = np.sum((img_gt == 0) & (im_mta == 0))
    n = np.sum(ske_img_gt_mta == 0)
    tnr = tn / n
    b_acc = (tpr + tnr) / 2
    precision = tp / np.sum(im_mta > 0)
    recall = tpr
    x_true, x_pred, y_true, y_pred = eval_splines(model, knots, points_gt, img_color.shape[:2])
    val_hd = hausdorff(x_true, x_pred, y_true, y_pred)
    mcd = mean_closest_distance(x_true, x_pred, y_true, y_pred)

    print('*' * 50)
    print('Performance')
    print('Hausdorff Distance: %02.4f' % (val_hd))
    print('MCD: %02.4f' % (mcd))
    print('bACC: %02.4f' % (b_acc))
    print('Precision: %02.4f' % (precision))
    print('Recall: %02.4f' % (recall))
    print('Time: %02.4f' % (time_elapsed))

lst_plt_imgs = [img_color, enh_img * 255, th_img * 255, dt_im * 255, im_comparative]
lst_plt_subtitles = ['Original', 'Enhanced', 'Vessel Segmentation', 'MTA Probability Map', 'Model vs. GT']
plot_other(lst_plt_imgs, lst_plt_subtitles, title='Robust MTA Detection-Modeling', ncols=len(lst_plt_imgs), nrows=1, size_col=6,
                   size_row=7)

cv2.imwrite(path_model, np.asarray(im_mta * 255, dtype=np.uint8))
print('Done.')
