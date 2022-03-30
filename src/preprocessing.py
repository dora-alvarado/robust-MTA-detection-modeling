import numpy as np
from skimage.morphology import black_tophat, disk, skeletonize
from skimage.filters import threshold_otsu
import cv2


def change_range(data, input_min, input_max, output_min, output_max):
    result = ((data - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min
    return result


def keep_largest_component(img):
    # img     : input image
    # min_size: minimum size of pixels we want to keep
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(np.asarray(img*255, dtype=np.uint8), connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1];
    nb_components = nb_components - 1
    if nb_components>1:
        max_idx = np.argmax(sizes)
    else:
        return img

    # your answer image
    mask = np.zeros((output.shape))
    mask[output == max_idx+1]=1
    return img*mask


def remove_small_components(img, min_size=100):
    # img     : input image
    # min_size: minimum size of pixels we want to keep
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(np.asarray(img*255, dtype=np.uint8), connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1];
    nb_components = nb_components - 1
    # your answer image
    mask = np.zeros((output.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            mask[output == i + 1] = 1
    return img*mask


def enhance_tophat(img, mask=None):
    if mask is None:
        mask = disk(9)
    img_tophat = black_tophat(img, mask)
    img_tophat = change_range(img_tophat, img_tophat.min(), img_tophat.max(), 0., 1.)
    return img_tophat


def thresholding_otsu(img, roi, min_size=500):
    th = threshold_otsu(img)
    th_img = img * roi > th
    th_img = remove_small_components(th_img, min_size=min_size)
    return th_img


def normalized_distance_transform(img, flag_erosion=True):
    dt_im = cv2.distanceTransform((img * 255).astype(np.uint8), cv2.DIST_L2, 5)
    dt_im /= dt_im.max()
    # remove small points
    mask_dt = dt_im != 0
    if flag_erosion:
        dt_im = dt_im * mask_dt
    return  dt_im

def get_gt_points(img_gt):
    # skeletonize groundtruth
    only_ske = skeletonize(img_gt)
    points_gt = np.argwhere(only_ske != 0)
    return points_gt


def otsu(h):
    N = np.sum(h)
    mB=0
    wB=0
    max_btwn_var = 0
    helperVec = range(256)

    prob = h / float(N)
    mu = np.sum(helperVec * prob)
    th=0
    for i in range(256):
        wB += prob[i]
        wF = 1-wB
        if (wB==0 or wF==0):
            continue
        mB +=i*prob[i]
        mF = (mu-mB)
        between = wB*wF*(mB/wB-mF/wF)**2
        if (between>max_btwn_var):
            th = i
            max_btwn_var = between

    return th
