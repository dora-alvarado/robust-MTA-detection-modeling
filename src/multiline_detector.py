import numpy as np
from skimage.morphology import diamond, disk
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.ndimage import rotate, convolve
import cv2


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


def fakepad(img, mask, erosionsize=5, iterations=50):
    nrows, ncols = mask.shape
    # erode mask to avoid weird region near the border
    se = disk(erosionsize)
    mask = binary_erosion(mask, se)
    dilated = img*mask
    oldmask = mask
    filter = np.ones((3,3))
    filter_coords = np.argwhere(filter > 0)-1
    se = diamond(1)
    for i in range(iterations):
        # finds pixels on the outer border
        newmask = binary_dilation(oldmask, se)
        outerborder = newmask & ~(oldmask)
        rows_cols = np.argwhere(outerborder)
        for j in range(len(rows_cols)):
            row, col = rows_cols[j]
            coords = [rows_cols[j]+px for px in filter_coords]
            coords = [c for c in coords if 0<=c[0]<nrows and 0<=c[1]<ncols]
            filtered = [dilated[c[0], c[1]] for c in coords if oldmask[c[0], c[1]]]
            try:
                dilated[row, col] = np.mean(filtered)
            except:
                dilated[row, col] = 0
        oldmask = newmask

    return dilated


def globalstandarize(img, mask):
    usedpixels = img[mask>0]
    m = np.mean(usedpixels)
    s = np.std(usedpixels)
    simg = np.zeros(img.shape)
    simg[mask > 0] = (usedpixels - m) / s
    return simg


def get_linemask(size):
    mask = np.zeros((size, size))
    halfsize = (size - 1) // 2
    mask[halfsize, :] = 1
    return mask


def get_lineresponse(img, w, L):
    avgresponse = cv2.blur(img, (w, w))
    linemask = get_linemask(L)
    bank = [rotate(linemask, theta, mode='nearest', reshape=False, order=0) for theta in range(0, 180, 15)]
    bank = [f/np.sum(f) for f in bank]
    responses = [convolve(img, f, mode='nearest')-avgresponse for f in bank]
    responses = np.asarray(responses)
    max_response = np.max(responses, axis=0)
    return max_response


def im_seg(img,mask,W=15,step=2, t = 0.56, min_area = 1000):
    img = img[:, :, 1]
    img = 1 - img
    img = fakepad(img, mask)
    Ls = range(1, W, step)

    features = [globalstandarize(get_lineresponse(img,W,L), mask) for L in Ls]
    features.append(globalstandarize(img, mask))
    features = np.asarray(features)
    segmentedimg = np.mean(features, axis=0)
    segmentedimg = segmentedimg>t
    segmentedimg = remove_small_components(segmentedimg, min_area)
    return segmentedimg

