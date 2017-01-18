import numpy as np

import cv2
from skimage.feature import peak_local_max
from skimage import measure

from collections import deque


def calc_space_scale_k(image, k=10, sigma=1.6,
                       borderType=cv2.BORDER_REPLICATE, **kwargs):
    if image.shape[0] % 2 == 0 or image.shape[1] % 2 == 0:
        raise NonOddException('Size of any image dimension must be odd!')
    blur_k = np.zeros((k + 1, image.shape[0], image.shape[1]))
    blur_k[0] = cv2.GaussianBlur(image, ksize=image.shape,
                                 sigmaX=1.6, borderType=cv2.BORDER_REPLICATE, **kwargs)
    for i in range(1, k + 1, 1):
        blur_k[i] = cv2.GaussianBlur(blur_k[i - 1], ksize=image.shape,
                                     sigmaX=1.6,
                                     borderType=cv2.BORDER_REPLICATE, **kwargs)
    return blur_k


def calc_DOG(datacube):
    dog_k = np.zeros((datacube.shape[0] - 1, datacube.shape[1], datacube.shape[2]))
    for i in range(0, datacube.shape[0] - 1, 1):
        dog_k[i] = datacube[i + 1] - datacube[i]
    return dog_k


def calc_LOG(datacube, ksize=None, depth=cv2.CV_64F):
    log_k = np.zeros((datacube.shape[0] - 1, datacube.shape[1], datacube.shape[2]))
    for i in range(0, datacube.shape[0] - 1, 1):
        log_k[i] = cv2.Laplacian(datacube[i], ddepth=depth)
    return log_k


def local_maxima(array, dist=10, num_peaks=np.inf, flat=False):
    coord = peak_local_max(array, min_distance=dist, num_peaks=num_peaks)
    if flat is True:
        coord = np.ravel_multi_index((coord[:, 0], coord[:, 1]), array.shape, mode='clip')
    return coord


def local_minima(array, dist=10, num_peaks=np.inf, flat=False):
    coord = peak_local_max(-array, min_distance=dist, num_peaks=num_peaks)
    if flat is True:
        coord = np.ravel_multi_index((coord[:, 0], coord[:, 1]), array.shape, mode='clip')
    return coord


def search_connections(point_indx, image, listofmax, nconnect=8):
    # p = point
    # indx = np.ravel_multi_index((p[0], p[1]), image.shape)
    image_flat = image.flatten()
    # maxarray = max_array.flatten()
    maxlen = image_flat.shape[0]
    state = np.zeros(maxlen, dtype='object')
    state[:] = 'undiscovered'
    state[point_indx] = 'discovered'

    offsets = get_offsets(image, nconnect=nconnect)
    Q = deque()
    Q.append(point_indx)

    connections = deque()

    while Q:
        v = Q.pop()
        for i in offsets:
            if (v + i) < maxlen:
                u = v + i
            else:
                continue
            if u in listofmax:
                connections.append(u)
            if state[u] == 'undiscovered':
                if image_flat[u] > image_flat[v]:
                    state[u] = 'discovered'
                    Q.append(u)
                # state[u] = 'discovered'

    return connections


def get_offsets(image, nconnect=8):
    rowlen = image.shape[1]
    if nconnect == 4:
        return [1, -1, rowlen, -rowlen]
    if nconnect == 8:
        return [1, -1, rowlen, -rowlen, rowlen + 1, -rowlen - 1, rowlen + 2, -rowlen - 2]


def count_convex_regions(image, target=1, sep=0, background=0):
    im = image.copy()
    im[np.where(im < sep)] = background
    im[np.where(im > sep)] = target
    labels, n = measure.label(im, return_num=True, background=background)
    return n
