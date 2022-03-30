import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


def change_range(data, input_min, input_max, output_min, output_max):
    result = ((data - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min
    return result


def plot_other(imgs, subtitles, title='', ncols=3, nrows=1, vmin=None, vmax=None, size_col=2,
               size_row=2.5):
    fig = plt.figure(figsize=(size_col * ncols, size_row * nrows))
    fig.suptitle(title, fontsize=28)
    nimgs = len(imgs)
    for i in range(nimgs):
        row = int(i / ncols)
        col = i % ncols
        ax = plt.subplot2grid((nrows, ncols), (row, col))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(subtitles[i], fontsize=24)
        ax.imshow(imgs[i], cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()


def tif2png(path_src, path_dst):
    listOfFiles_src = os.listdir(path_src)
    for file in listOfFiles_src:
        # read image and gt
        img = Image.open(os.path.join(path_src, file))
        img.save(os.path.join(path_dst, file[:-3]+'png'), 'png')


def read_grayscale_img(path):
    img = Image.open(path).convert('L')
    m, n = img.getdata().size
    img = np.asarray(img.getdata(), dtype=np.uint8).reshape(n, m) / 255.
    return img


def read_color_img(path):
    img = Image.open(path)
    m, n = img.getdata().size
    img = np.asarray(img.getdata(), dtype=np.uint8).reshape(n, m, 3) / 255.
    return img


def save_numpy_as_img(img, path):
    img = change_range(img, 0., 1., 0., 255.)
    img = Image.fromarray(img).convert('RGB')
    img.save(path)

