"""
Critical Net
=====

# An open-source Python implementation of the critical net algorytm based on
"Gu S., Zheng Y., Tomasi C. (2010) Critical Nets and Beta-Stable Features for Image Matching.
In: Daniilidis K., Maragos P., Paragios N. (eds) Computer Vision – ECCV
2010. ECCV 2010. Lecture Notes in Computer Science, vol 6313. Springer,
Berlin, Heidelberg"


Web Links
---------
http://link.springer.com/chapter/10.1007/978-3-642-15558-1_48

.. note::
    "Gu S., Zheng Y., Tomasi C. (2010) Critical Nets and Beta-Stable Features for Image Matching.
    In: Daniilidis K., Maragos P., Paragios N. (eds) Computer Vision – ECCV 2010. ECCV 2010. Lecture Notes in Computer Science, vol 6313. Springer, Berlin, Heidelberg"
    .. _a link: http://link.springer.com/chapter/10.1007/978-3-642-15558-1_48



.. module:: criticalnet
   :platform: Unix, Windows
   :synopsis: An implementation of critical net computing algorytm


"""
import numpy as np
from matplotlib import pyplot as plt

import cv2

import networkx as nx

from .util import *


class CriticalNet:

    def __init__(self, image=None, ktimes=10, kernel='gauss', sigma=1.6, border='replicate',
                 pixel_connect=8, lap_mode='DOG',  graph_toolkit=None):
        """
        Critical net class

        Parameters
        ----------
        image : 2d numpy array of odd dimensions
            original image
        ktimes : int
            maximal level of space-scale representation. Default is 10.
        kernel: str
            name of kernel using for space-scale representation.
        sigma: float
            value of sigma using for space-scale representation.
        border: str
            method of handling borders in space scaling.
        pixel_connect: int, 4|8
            pixel connectivity using for critical net calculation.

        Examples
        ---------
        cri_net = CriticalNet(im=image, ktimes=101)
        cri_net.calc_sscale()
        cri_net.calc_lap('LOG')
        cri_net.calc_beta_levels()
        cri_net.calc_cnet(beta=10, extrema_dist=15)
        cri_net.draw()
        --------------------
        cri_net = CriticalNet(im=image, ktimes=101)
        cri_net.compute(beta=10, draw=True)

        """
        self.border_dict = {'replicate': cv2.BORDER_REPLICATE,
                            'constant': cv2.BORDER_CONSTANT,
                            'reflect': cv2.BORDER_REFLECT}  #: see OpenCV:BorderTypes

        self.im = auto_padding(image)  # : original image
        self.ktimes = ktimes
        self.kernel = kernel
        self.sigma = sigma
        self.border = self.border_dict[border]
        self.nconnect = pixel_connect
        self.lap_mode = lap_mode

        self.G = None  # : graph of critical net (networkx.DiGraph)
        self.sscale = None  # : 3d array of space-scale representation of original image
        self.lap = None  # : 3d array of laplacian of "space-scale2" array
        self.blev = None  # : list of beta levels for laplacian image
        self.nconvex = None  # : list of number of separate regions where laplacian image > 0
        self.current_beta = None

    def __call__(self):
        return self.G

    def calc_sscale(self, ktimes=None, **kwargs):
        if ktimes is None:
            ktimes = self.ktimes
        sigma = kwargs.pop('sigma', self.sigma)
        border = kwargs.pop('borderType', self.border)
        if self.kernel == 'gauss':
            self.sscale = calc_space_scale_k(self.im, ktimes, sigma, border, **kwargs)

    def calc_lap(self, lap_mode=None, **kwargs):
        if not lap_mode:
            lap_mode = self.lap_mode
        if lap_mode == 'DOG':
            self.lap = calc_DOG(self.sscale)
        if lap_mode == 'LOG':
            self.lap = calc_LOG(self.sscale, **kwargs)

    def calc_cnet(self, beta=10, extrema_dist=10, num_peaks=np.inf, full=False, k=None):
        if not k:
            k = self.scale_from_beta(beta)
            self.current_beta = beta
        print('Scale: ', k)
        im = self.lap[k]
        self.G = nx.DiGraph()

        coord_max = local_maxima(im, dist=extrema_dist, num_peaks=num_peaks, flat=True)
        coord_min = local_minima(im, dist=extrema_dist, num_peaks=num_peaks, flat=True)
        for m in coord_min:
            m_tuple = np.unravel_index(m, im.shape)
            d = search_connections(m, im, coord_max, self.nconnect)
            if d:
                self.G.add_node(m_tuple)
            for n in d:
                n_tuple = np.unravel_index(n, im.shape)
                self.G.add_node(n_tuple)
                self.G.add_edge(m_tuple, n_tuple)

    def beta_from_scale(self, k):
        return self.blev[k]

    def scale_from_beta(self, beta):
        kscale = np.where(self.blev == beta)
        if kscale[0].size == 0:
            raise BetaException('Beta level of ' + str(beta) + ' doesn\'t exist!')
        return np.min(kscale)

    def calc_beta_levels(self):
        nlap = self.lap.shape[0]
        nconvex = np.zeros(nlap, dtype='int32')
        blev = np.zeros(nlap, dtype='int32')

        # Connected-component labeling
        # Counting of number of separate regions where Laplacian > 0
        for j in range(nlap):
            im = self.lap[j]
            ncount = count_convex_regions(im, target=1, sep=0, background=0)
            nconvex[j] = ncount

        # Beta level calculation
        # Level k is beta-stable if k is smallest integer (given value Beta) for which the value of nconvex[k]
        # does not change
        # in range nconvex[k-beta:k]
        # e.g. lap[25] is 10-stable if all points of the nconvex[15:26] slice have the same value
        bcount = 0
        prev_n = nconvex[0]
        for j in range(1, nlap, 1):
            n = nconvex[j]
            if n == prev_n:
                bcount = bcount + 1
            else:
                bcount = 0
            blev[j] = bcount
            prev_n = n
        self.blev = blev
        self.nconvex = nconvex
        self.print_beta_levels()

    def print_beta_levels(self):
        blev = self.blev
        n = blev.max()
        print('Beta  | Scale_k:')
        print('-----------------------------')
        for j in range(n + 1):
            kscale = np.min(np.where(blev == j))
            print(j, '       ', kscale)

    def get_cnet_pos(self):
        pos_d = dict()

        for node in self.G.nodes():
            pos_d[node] = [node[1], node[0]]
        return pos_d

    def draw(self, im=None):
        if im is None:
            im = self.im
        pos = self.get_cnet_pos()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(im)
        nx.draw(self.G, pos=pos, ax=ax, node_size=20, width=0.5, edge_color='yellow')
        plt.show()

    def imread2(self, path_to_image):
        self.im = imread(path_to_image, mode='F')

    def compute(self, beta=10, extrema_dist=10, num_peaks=np.inf, full=False, k=None, draw=True, force=False):
        if self.sscale is None or force:
            self.calc_sscale()
        if self.lap is None or force:
            self.calc_lap()
        if self.blev is None or force:
            self.calc_beta_levels()
        self.calc_cnet(beta=beta, extrema_dist=extrema_dist, num_peaks=num_peaks,
                       full=False, k=None)
        if draw:
            self.draw()


class BetaException(Exception):
    # Raise if the beta level not found
    pass


class NonOddException(Exception):
    # Raise if image dimensions is not odd
    pass
