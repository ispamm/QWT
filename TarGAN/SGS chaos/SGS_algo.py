import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from skimage.io import imread
import pywt
import pywt.data
from PIL import Image
from scipy.fftpack import hilbert as ht
from six.moves import xrange
import torchvision
from PIL import Image, ImageOps
from scipy.stats import gaussian_kde
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from collections import Counter
from tqdm import tqdm
from nltk.cluster.util import VectorSpaceClusterer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import copy
import random
import sys

from dataset_dicom_chaos import ChaosDataset_Syn_new



import os

class QWavelet(object):
    def __init__(self):
        self.r = torchvision.transforms.Resize(size=(512, 512))
        self.frompil_totensor = torchvision.transforms.ToTensor()
        return


    def pywt_coeffs(self):
        # coefficients from pywt db8 (http://wavelets.pybytes.com/wavelet/db8/)
        gl = [-0.00011747678400228192,
        0.0006754494059985568,
        -0.0003917403729959771,
        -0.00487035299301066,
        0.008746094047015655,
        0.013981027917015516,
        -0.04408825393106472,
        -0.01736930100202211,
        0.128747426620186,
        0.00047248457399797254,
        -0.2840155429624281,
        -0.015829105256023893,
        0.5853546836548691,
        0.6756307362980128,
        0.3128715909144659,
        0.05441584224308161
        ]

        gh = [-0.05441584224308161,
        0.3128715909144659,
        -0.6756307362980128,
        0.5853546836548691,
        0.015829105256023893,
        -0.2840155429624281,
        -0.00047248457399797254,
        0.128747426620186,
        0.01736930100202211,
        -0.04408825393106472,
        -0.013981027917015516,
        0.008746094047015655,
        0.00487035299301066,
        -0.0003917403729959771,
        -0.0006754494059985568,
        -0.00011747678400228192
        ]
        return np.asarray(gl), np.asarray(gh)

    # Compute Hilbert transform of the filters G_L and G_H
    def get_hilbert_filters(self, gl, gh):
        fl = ht(gl)
        fh = ht(gh)
        return fl, fh

    def get_filters(self):
        gl, gh = self.pywt_coeffs()
        fl, fh = self.get_hilbert_filters(gl, gh)
        return gl, gh, fl, fh

    def reflect(self, x, minx, maxx):
        """Reflect the values in matrix *x* about the scalar values *minx* and
        *maxx*.  Hence a vector *x* containing a long linearly increasing series is
        converted into a waveform which ramps linearly up and down between *minx* and
        *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers + 0.5), the
        ramps will have repeated max and min samples.
        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.
        """
        x = np.asanyarray(x)
        rng = maxx - minx
        rng_by_2 = 2 * rng
        mod = np.fmod(x - minx, rng_by_2)
        normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
        out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
        return np.array(out, dtype=x.dtype)

    def as_column_vector(self, v):
        """Return *v* as a column vector with shape (N,1).
        """
        v = np.atleast_2d(v)
        if v.shape[0] == 1:
            return v.T
        else:
            return v

    def _centered(self, arr, newsize):
        # Return the center newsize portion of the array.
        # (Shamelessly cribbed from scipy.)
        newsize = np.asanyarray(newsize)
        currsize = np.array(arr.shape)
        startind = (currsize - newsize) // 2
        endind = startind + newsize
        myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
        return arr[tuple(myslice)]

    def _column_convolve(self, X, h):
        """Convolve the columns of *X* with *h* returning only the 'valid' section,
        i.e. those values unaffected by zero padding. Irrespective of the ftype of
        *h*, the output will have the dtype of *X* appropriately expanded to a
        floating point type if necessary.
        We assume that h is small and so direct convolution is the most efficient.
        """
        Xshape = np.asanyarray(X.shape)
        h = h.flatten().astype(X.dtype)
        h_size = h.shape[0]

        full_size = X.shape[0] + h_size - 1
        Xshape[0] = full_size

        out = np.zeros(Xshape, dtype=X.dtype)
        for idx in xrange(h_size):
            out[idx:(idx+X.shape[0]),...] += X * h[idx]

        outShape = Xshape.copy()
        outShape[0] = abs(X.shape[0] - h_size) + 1
        return self._centered(out, outShape)

    def colfilter(self, X, h):
        """Filter the columns of image *X* using filter vector *h*, without decimation.
        If len(h) is odd, each output sample is aligned with each input sample
        and *Y* is the same size as *X*.  If len(h) is even, each output sample is
        aligned with the mid point of each pair of input samples, and Y.shape =
        X.shape + [1 0].
        :param X: an image whose columns are to be filtered
        :param h: the filter coefficients.
        :returns Y: the filtered image.
        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, August 2013
        .. codeauthor:: Cian Shaffrey, Cambridge University, August 2000
        .. codeauthor:: Nick Kingsbury, Cambridge University, August 2000
        """

        # Interpret all inputs as arrays
    #     X = asfarray(X)
        X = np.asarray(X)
        h = self.as_column_vector(h)
        # print("X shape:", X.shape)
#         c, r, width = X.shape
        c, r = X.shape
        m = h.shape[0]
        m2 = np.fix(m*0.5)

        # Symmetrically extend with repeat of end samples.
        # Use 'reflect' so r < m2 works OK.
        xe = self.reflect(np.arange(-m2, r+m2, dtype=np.int), -0.5, r-0.5)

        # Perform filtering on the columns of the extended matrix X(xe,:), keeping
        # only the 'valid' output samples, so Y is the same size as X if m is odd.
        Y = self._column_convolve(X[xe,:], h)

        return Y

    def qwt(self, image, gl, gh, fl, fh, only_low=True, quad=1):
        '''Compute the QWT. Just compute the low frequency coefficients.
        Return L_G L_G, L_F L_G, L_G L_F, L_F L_F.'''

        if only_low:
            t1 = self.colfilter(image, gl)
            # t1 = self.downsample(t1)
            lglg = self.colfilter(t1, gl)
            t2 = self.colfilter(image, fl)
            # t2 = self.downsample(t2)
            lflg = self.colfilter(t2, gl)
            t3 = self.colfilter(image, gl)
            # t3 = self.downsample(t3)
            lglf = self.colfilter(t3, fl)
            t4 = self.colfilter(image, fl)
            # t4 = self.downsample(t4)
            lflf = self.colfilter(t4, fl)
            # lglg, lflg, lglf, lflf = t1, t2, t3, t4
            # lglg, lflg, lglf, lflf = full_quat_downsample(lglg, lflg, lglf, lflf)

            return lglg, lflg, lglf, lflf
        else:
            if quad==1:
                t1 = self.colfilter(image, gl)
                lglg = self.colfilter(t1, gl)
                t2 = self.colfilter(image, gl)
                lghg = self.colfilter(t2, gh)
                t3 = self.colfilter(image, gh)
                hglg = self.colfilter(t3, gl)
                t4 = self.colfilter(image, gh)
                hghg = self.colfilter(t4, gh)

                return lglg, lghg, hglg, hghg

            elif quad==2:
                t1 = self.colfilter(image, fl)
                lflg = self.colfilter(t1, gl)
                t2 = self.colfilter(image, fl)
                lfhg = self.colfilter(t2, gh)
                t3 = self.colfilter(image, fh)
                hflg = self.colfilter(t3, gl)
                t4 = self.colfilter(image, fh)
                hfhg = self.colfilter(t4, gh)

                return lflg, lfhg, hflg, hfhg

            elif quad==3:
                t1 = self.colfilter(image, gl)
                lglf = self.colfilter(t1, fl)
                t2 = self.colfilter(image, gl)
                lghf = self.colfilter(t2, fh)
                t3 = self.colfilter(image, gh)
                hglf = self.colfilter(t3, fl)
                t4 = self.colfilter(image, gh)
                hghf = self.colfilter(t4, fh)

                return lglf, lghf, hglf, hghf
            
            elif quad==4:
                t1 = self.colfilter(image, fl)
                lflf = self.colfilter(t1, fl)
                t2 = self.colfilter(image, fl)
                lfhf = self.colfilter(t2, fh)
                t3 = self.colfilter(image, fh)
                hflf = self.colfilter(t3, fl)
                t4 = self.colfilter(image, fh)
                hfhf = self.colfilter(t4, fh)

                return lflf, lfhf, hflf, hfhf
            
            elif quad=="all":
                t1 = self.colfilter(image, gl)
                t1 = self.downsample(t1)
                lglg = self.colfilter(t1, gl)
                t2 = self.colfilter(image, gl)
                t2 = self.downsample(t2)
                lghg = self.colfilter(t2, gh)
                t3 = self.colfilter(image, gh)
                t3 = self.downsample(t3)
                hglg = self.colfilter(t3, gl)
                t4 = self.colfilter(image, gh)
                t4 = self.downsample(t4)
                hghg = self.colfilter(t4, gh)

                t1 = self.colfilter(image, fl)
                t1 = self.downsample(t1)
                lflg = self.colfilter(t1, gl)
                t2 = self.colfilter(image, fl)
                t2 = self.downsample(t2)
                lfhg = self.colfilter(t2, gh)
                t3 = self.colfilter(image, fh)
                t3 = self.downsample(t3)
                hflg = self.colfilter(t3, gl)
                t4 = self.colfilter(image, fh)
                t4 = self.downsample(t4)
                hfhg = self.colfilter(t4, gh)

                t1 = self.colfilter(image, gl)
                t1 = self.downsample(t1)
                lglf = self.colfilter(t1, fl)
                t2 = self.colfilter(image, gl)
                t2 = self.downsample(t2)
                lghf = self.colfilter(t2, fh)
                t3 = self.colfilter(image, gh)
                t3 = self.downsample(t3)
                hglf = self.colfilter(t3, fl)
                t4 = self.colfilter(image, gh)
                t4 = self.downsample(t4)
                hghf = self.colfilter(t4, fh)

                t1 = self.colfilter(image, fl)
                t1 = self.downsample(t1)
                lflf = self.colfilter(t1, fl)
                t2 = self.colfilter(image, fl)
                t2 = self.downsample(t2)
                lfhf = self.colfilter(t2, fh)
                t3 = self.colfilter(image, fh)
                t3 = self.downsample(t3)
                hflf = self.colfilter(t3, fl)
                t4 = self.colfilter(image, fh)
                t4 = self.downsample(t4)
                hfhf = self.colfilter(t4, fh)

                # Mean of components
                # ll = (lglg + lflg + lglf + lflf)/2
                # lh = (lghg + lfhg + lghf + lfhf)/2
                # hl = (hglg + hflg + hglf + hflf)/2
                # hh = (hghg + hfhg + hghf + hfhf)/2

                ll = (lflg + lglf + lflf)/3
                lh = (lfhg + lghf + lfhf)/3
                hl = (hflg + hglf + hflf)/3
                hh = (hfhg + hghf + hfhf)/3

                return (ll, lh, hl, hh), (lglg, lghg, hglg, hghg, lflg, lfhg, hflg, hfhg, lglf, lghf, hglf, hghf, lflf, lfhf, hflf, hfhf)

    def quat_mag_and_phase(self, q0, q1, q2, q3):
        '''Compute the magnitude and phase quaternion representation.'''
        q_mp = np.asarray([q0, q1, q2, q3])

        phi = np.arctan(2*(q0+q1*q3)/(q0**2+q1**2-q2**2-q3**2))
        theta = np.arctan((q0*q1+q2*q3)/(q0**2-q1**2+q2**2-q3**2))
        psi = 1/2*np.arctan(2*(q0*q3-q3*q1))

        phi = np.nan_to_num(phi)
        theta = np.nan_to_num(theta)
        psi = np.nan_to_num(psi)

        q0_mag = np.linalg.norm(q_mp, axis=0, ord=2)
        q1_phase = np.exp(phi)
        q2_phase = np.exp(theta)
        q3_phase = np.exp(psi)

        return q0_mag, q1_phase, q2_phase, q3_phase

    def downsample(self, component):
        return component[::2, ::2]

    def full_quat_downsample(self, q0, q1, q2, q3):
        q0 = self.downsample(q0)
        q1 = self.downsample(q1)
        q2 = self.downsample(q2)
        q3 = self.downsample(q3)
        return q0, q1, q2, q3

    def __call__(self, input):
        
        img = input
#         img = ImageOps.grayscale(input)
#         img = self.r(img)
        img = self.frompil_totensor(img)
        img = img.squeeze(0)
        # print("image shape:", img.shape)
#         img = img.reshape(512, 512)
        gl, gh, fl, fh = self.get_filters()
#         q0, q1, q2, q3 = self.qwt(img, gl, gh, fl, fh, only_low=False, quad="all")
        # q0, q1, q2, q3 = self.quat_mag_and_phase(q0, q1, q2, q3)
#         q0, q1, q2, q3 = q0[1:,:], q1[1:,:], q2[1:,:], q3[1:,:]
#         q0 = np.expand_dims(q0, axis=0)
#         q1 = np.expand_dims(q1, axis=0)
#         q2 = np.expand_dims(q2, axis=0)
#         q3 = np.expand_dims(q3, axis=0)

#         image = np.concatenate((q0, q1, q2, q3), axis=0)
#         print(q0.shape)
#         image = np.stack((q0, q1, q2, q3), axis=0)

        ele, all_ = self.qwt(img, gl, gh, fl, fh, only_low=False, quad="all")
        subbands = []
        for subband in all_:
            subband = subband[1:,:]
#             subband = np.expand_dims(subband, axis=2)
#             print(subband.shape)
            subbands.append(subband)

#         return image
        return subbands



#################
#### Filters ####
#################

def pywt_coeffs():
    # coefficients from pywt db8 (http://wavelets.pybytes.com/wavelet/db8/)
    gl = [-0.00011747678400228192,
    0.0006754494059985568,
    -0.0003917403729959771,
    -0.00487035299301066,
    0.008746094047015655,
    0.013981027917015516,
    -0.04408825393106472,
    -0.01736930100202211,
    0.128747426620186,
    0.00047248457399797254,
    -0.2840155429624281,
    -0.015829105256023893,
    0.5853546836548691,
    0.6756307362980128,
    0.3128715909144659,
    0.05441584224308161
    ]

    gh = [-0.05441584224308161,
    0.3128715909144659,
    -0.6756307362980128,
    0.5853546836548691,
    0.015829105256023893,
    -0.2840155429624281,
    -0.00047248457399797254,
    0.128747426620186,
    0.01736930100202211,
    -0.04408825393106472,
    -0.013981027917015516,
    0.008746094047015655,
    0.00487035299301066,
    -0.0003917403729959771,
    -0.0006754494059985568,
    -0.00011747678400228192
    ]
    return np.asarray(gl), np.asarray(gh)

# Compute Hilbert transform of the filters G_L and G_H
def get_hilbert_filters(gl, gh):
    fl = ht(gl)
    fh = ht(gh)
    return fl, fh

def get_filters():
    gl, gh = pywt_coeffs()
    fl, fh = get_hilbert_filters(gl, gh)
    return gl, gh, fl, fh

#################
#### STEP 1 #####
#################


def compute_energy(subband, filters, kind="pixels"):
    if kind == "pixels":
        return sum(sum(subband**2))
    elif kind == "pix+coeff":
        gl, gh, fl, fh = get_filters()
        if filters == "gg":
            return sum(sum(gl*gh))*sum(sum(subband**2))
        elif filters == "ff":
            return sum(sum(fl*fh))*sum(sum(subband**2))
        elif filters == "lglf" or filters == "lflg":
            return sum(sum(gl*fl))*sum(sum(subband**2))
        elif filters == "hglf" or filters == "lfhg":
            return sum(sum(gh*fl))*sum(sum(subband**2))
        elif filters == "lghf" or filters == "lfhg":
            return sum(sum(gl*fh))*sum(sum(subband**2))
        elif filters == "hghf" or filters == "hfhg":
            return sum(sum(gh*fh))*sum(sum(subband**2))




def step1(images):
    '''
    Compute energy for each sub-band.
    '''

    lglg = []
    lghg = []
    hglg = []
    hghg = []
    lglf = []
    lghf = []
    hglf = []
    hghf = []
    lflg = []
    lfhg = []
    hflg = []
    hfhg = []
    lflf = []
    lfhf = []
    hflf = []
    hfhf = []
    all_energies = []

    for qwim in images:
        counter = 1
        for subband in qwim:
            if counter == 1:
                energy = compute_energy(subband, filters="gg")
                lglg.append(energy)
            elif counter == 2:
                energy = compute_energy(subband, filters="gg")
                lghg.append(energy)
            elif counter == 3:
                energy = compute_energy(subband, filters="gg")
                hglg.append(energy)
            elif counter == 4:
                energy = compute_energy(subband, filters="gg")
                hghg.append(energy)
            elif counter == 5:
                energy = compute_energy(subband, filters="lglf")
                lglf.append(energy)
            elif counter == 6:
                energy = compute_energy(subband, filters="lghf")
                lghf.append(energy)
            elif counter == 7:
                energy = compute_energy(subband, filters="hglf")
                hglf.append(energy)
            elif counter == 8:
                energy = compute_energy(subband, filters="hghf")
                hghf.append(energy)
            elif counter == 9:
                energy = compute_energy(subband, filters="lflg")
                lflg.append(energy)
            elif counter == 10:
                energy = compute_energy(subband, filters="lfhg")
                lfhg.append(energy)
            elif counter == 11:
                energy = compute_energy(subband, filters="hflg")
                hflg.append(energy)
            elif counter == 12:
                energy = compute_energy(subband, filters="hfhg")
                hfhg.append(energy)
            elif counter == 13:
                energy = compute_energy(subband, filters="ff")
                lflf.append(energy)
            elif counter == 14:
                energy = compute_energy(subband, filters="ff")
                lfhf.append(energy)
            elif counter == 15:
                energy = compute_energy(subband, filters="ff")
                hflf.append(energy)
            elif counter == 16:
                energy = compute_energy(subband, filters="ff")
                hfhf.append(energy)
            
            # List of all energies
            all_energies.append(energy)
                
            counter += 1
    return all_energies, (lglg, lghg, hglg, hghg, lflg, lfhg, hflg, hfhg, lglf, lghf, hglf, hghf, lflf, lfhf, hflf, hfhf)

#################
#### STEP 2 #####
#################


def get_pdf(lst):
    #     kde = gaussian_kde(all_energies)
    kde = gaussian_kde(lst)
    pdf = kde.pdf(lst)
    return pdf

def step2(lst):
    lglg, lghg, hglg, hghg, lflg, lfhg, hflg, hfhg, lglf, lghf, hglf, hghf, lflf, lfhf, hflf, hfhf = lst

    lglg_pdf = get_pdf(lglg)
    lghg_pdf = get_pdf(lghg)
    hglg_pdf = get_pdf(hglg)
    hghg_pdf = get_pdf(hghg)
    lglf_pdf = get_pdf(lglf)
    lghf_pdf = get_pdf(lghf)
    hglf_pdf = get_pdf(hglf)
    hghf_pdf = get_pdf(hglf)
    lflg_pdf = get_pdf(lflg)
    lfhg_pdf = get_pdf(lfhg)
    hflg_pdf = get_pdf(hflg)
    hfhg_pdf = get_pdf(hfhg)
    lflf_pdf = get_pdf(lflf)
    lfhf_pdf = get_pdf(lfhf)
    hflf_pdf = get_pdf(hflf)
    hfhf_pdf = get_pdf(hfhf)
    joint_pdfs = np.stack([lglg_pdf, lghg_pdf, hglg_pdf, hghg_pdf, lglf_pdf, lghf_pdf, hglf_pdf, hghf_pdf, lflg_pdf, lfhg_pdf, hflg_pdf, hfhg_pdf, lflf_pdf, lfhf_pdf, hflf_pdf, hfhf_pdf], axis=0)
    return joint_pdfs


#################
#### STEP 3 #####
#################


def mutual_information(px, pt):
    pxt = np.histogram2d(px, pt)[0]
#     print(pxt)
    return mutual_info_score(None, None, contingency=pxt)
#     return normalized_mutual_info_score(px, pt)

#################
#### STEP 4 #####
#################


class KMeansClusterer(VectorSpaceClusterer):
    """
    The K-means clusterer starts with k arbitrary chosen means then allocates
    each vector to the cluster with the closest mean. It then recalculates the
    means of each cluster as the centroid of the vectors in the cluster. This
    process repeats until the cluster memberships stabilise. This is a
    hill-climbing algorithm which may converge to a local maximum. Hence the
    clustering is often repeated with random initial means and the most
    commonly occurring output means are chosen.
    """

    def __init__(
        self,
        num_means,
        distance,
        repeats=1,
        conv_test=1e-6,
        initial_means=None,
        normalise=False,
        svd_dimensions=None,
        rng=None,
        avoid_empty_clusters=False,
    ):

        """
        :param  num_means:  the number of means to use (may use fewer)
        :type   num_means:  int
        :param  distance:   measure of distance between two vectors
        :type   distance:   function taking two vectors and returning a float
        :param  repeats:    number of randomised clustering trials to use
        :type   repeats:    int
        :param  conv_test:  maximum variation in mean differences before
                            deemed convergent
        :type   conv_test:  number
        :param  initial_means: set of k initial means
        :type   initial_means: sequence of vectors
        :param  normalise:  should vectors be normalised to length 1
        :type   normalise:  boolean
        :param svd_dimensions: number of dimensions to use in reducing vector
                               dimensionsionality with SVD
        :type svd_dimensions: int
        :param  rng:        random number generator (or None)
        :type   rng:        Random
        :param avoid_empty_clusters: include current centroid in computation
                                     of next one; avoids undefined behavior
                                     when clusters become empty
        :type avoid_empty_clusters: boolean
        """
        VectorSpaceClusterer.__init__(self, normalise, svd_dimensions)
        self._num_means = num_means
        self._distance = distance
        self._max_difference = conv_test
        assert not initial_means or len(initial_means) == num_means
        self._means = initial_means
        assert repeats >= 1
        assert not (initial_means and repeats > 1)
        self._repeats = repeats
        self._rng = rng if rng else random.Random()
        self._avoid_empty_clusters = avoid_empty_clusters


    def cluster_vectorspace(self, vectors, trace=False):
        if self._means and self._repeats > 1:
            print("Warning: means will be discarded for subsequent trials")

        meanss = []
        for trial in range(self._repeats):
            if trace:
                print("k-means trial", trial)
            if not self._means or trial > 1:
                self._means = self._rng.sample(list(vectors), self._num_means)
            self._cluster_vectorspace(vectors, trace)
            meanss.append(self._means)

        if len(meanss) > 1:
            # sort the means first (so that different cluster numbering won't
            # effect the distance comparison)
            for means in meanss:
                means.sort(key=sum)

            # find the set of means that's minimally different from the others
            min_difference = min_means = None
            for i in range(len(meanss)):
                d = 0
                for j in range(len(meanss)):
                    if i != j:
                        d += self._sum_distances(meanss[i], meanss[j])
#                 if min_difference is None or d < min_difference:
                if min_difference is None or d > min_difference:
                    min_difference, min_means = d, meanss[i]

            # use the best means
            self._means = min_means


    def _cluster_vectorspace(self, vectors, trace=False):
        if self._num_means < len(vectors):
            # perform k-means clustering
            converged = False
            while not converged:
                # assign the tokens to clusters based on minimum distance to
                # the cluster means
                clusters = [[] for m in range(self._num_means)]
                for vector in vectors:
                    index = self.classify_vectorspace(vector)
                    clusters[index].append(vector)
#                     print(index)

                if trace:
                    print("iteration")
                # for i in range(self._num_means):
                # print '  mean', i, 'allocated', len(clusters[i]), 'vectors'

                # recalculate cluster means by computing the centroid of each cluster
#                 print("centroids:")
#                 print(self._centroid)
#                 print()
#                 print("clusters:")
#                 print(len(clusters))
                new_means = list(map(self._centroid, clusters, self._means))

                # measure the degree of change from the previous step for convergence
                difference = self._sum_distances(self._means, new_means)
#                 if difference < self._max_difference:
#                 print("Difference:", difference)
                if difference > self._max_difference:
                    converged = True

                # remember the new means
                self._means = new_means

    def classify_vectorspace(self, vector):
        # finds the closest cluster centroid
        # returns that cluster's index
        best_distance = best_index = None
        for index in range(len(self._means)):
            mean = self._means[index]
            dist = self._distance(vector, mean)
#             print("Vect/mean:", vector, mean)
#             print("Dist value:", dist)
#             if best_distance is None or dist < best_distance:
            if best_distance is None or dist > best_distance:
                best_index, best_distance = index, dist
        return best_index


    def num_clusters(self):
        if self._means:
            return len(self._means)
        else:
            return self._num_means


    def means(self):
        """
        The means used for clustering.
        """
        return self._means


    def _sum_distances(self, vectors1, vectors2):
        difference = 0.0
        for u, v in zip(vectors1, vectors2):
            difference += self._distance(u, v)
        return difference

    def _centroid(self, cluster, mean):
        if self._avoid_empty_clusters:
            centroid = copy.copy(mean)
            for vector in cluster:
                centroid += vector
            return centroid / (1 + len(cluster))
        else:
            if not len(cluster):
                sys.stderr.write("Error: no centroid defined for empty cluster.\n")
                sys.stderr.write(
                    "Try setting argument 'avoid_empty_clusters' to True\n"
                )
                assert False
            centroid = copy.copy(cluster[0])
            for vector in cluster[1:]:
                centroid += vector
            return centroid / len(cluster)

    def __repr__(self):
        return "<KMeansClusterer means=%s repeats=%d>" % (self._means, self._repeats)



def get_max_energy(lst):
    return np.max(lst)

def get_subband_with_max_energy(lst, subbands):
    max_energy = get_max_energy(lst)
    for idx in range(len(subbands)):
        if max_energy in subbands[idx]:
#             print("Found max energy in subband #:", idx)
            return idx

def repeat_kmeans(joint_pdfs, subbands, ntimes):
    
    max_energy_subbands = []
    
    for rep in tqdm(range(ntimes)):
        
        # Apply kmeans
        centers = kmeans_plusplus_initializer(joint_pdfs, 4).initialize()
        clusterer = KMeansClusterer(4, mutual_information, conv_test=6.3, initial_means=centers)
        clusters = clusterer.cluster(joint_pdfs, True)
        
        # Build subbands dictionary
        subbands_dict = dict()
        for idx in range(len(clusters)):
            if str(clusters[idx]) not in subbands_dict.keys():
                subbands_dict[str(clusters[idx])] = list(subbands[idx])
            elif str(clusters[idx]) in subbands_dict.keys():
                subbands_dict[str(clusters[idx])] += subbands[idx]

        val0 = get_subband_with_max_energy(subbands_dict["0"], subbands)
        val1 = get_subband_with_max_energy(subbands_dict["1"], subbands)
        val2 = get_subband_with_max_energy(subbands_dict["2"], subbands)
        val3 = get_subband_with_max_energy(subbands_dict["3"], subbands)
        max_energy_subbands.append(val0)
        max_energy_subbands.append(val1)        
        max_energy_subbands.append(val2)
        max_energy_subbands.append(val3)
        
    return Counter(max_energy_subbands)


def main(dataset, kmeans_repetitions):
    qw = QWavelet()

    if dataset == "kvasir":
        path = "G:\Il mio Drive\Dottorato\Tesisti\Carnemolla\CODICE\Segmentation task\Datasets\Kvasir_im_mask\images"
        files = os.listdir(path)
    elif dataset == "ixi":
        path = "G:/Il mio Drive/Dottorato/Tesisti/Carnemolla/CODICE/Reconstrunction task/IXI_preprocess_dataset"
        path1 = path + "/IXI-T1/T1.npz"
        path2 = path + "/IXI-T2/T2.npz"        
        files1 = np.load(path1, allow_pickle=True)['arr_0'] # array of uint8 (581, 256,256) valori [0,255]
        files2 = np.load(path2, allow_pickle=True)['arr_0'] # array of uint8 (578, 256,256) valori [0,255]
        files = np.concatenate((files1, files2), axis=0)

    elif dataset == "kmnist":
        path = "G:/Il mio Drive/Dottorato/Tesisti/Carnemolla/CODICE/Reconstrunction task/K-MNIST"
        path1 = path+"/kmnist-train-imgs.npz"
        files = np.load(path1, allow_pickle=True)['arr_0'] # array of uint8 (60000,28,28) for train # array of uint8 (10000,28,28) for test

    elif dataset == "chaos":
        path = "G:/Il mio Drive/Dottorato/Tesisti/Carnemolla/CODICE/Generation task/chaos2019/png8020/train"
        path1 = "G:/Il mio Drive/Dottorato/Tesisti/Carnemolla/CODICE/Generation task/chaos2019/png8020/train/ct"
        path2 = "G:/Il mio Drive/Dottorato/Tesisti/Carnemolla/CODICE/Generation task/chaos2019/png8020/train/t1"
        path3 = "G:/Il mio Drive/Dottorato/Tesisti/Carnemolla/CODICE/Generation task/chaos2019/png8020/train/t2"
        files1 = os.listdir(path1)
        files2 = os.listdir(path2)
        files3 = os.listdir(path3)
        files1 = [os.path.join(path1, file) for file in files1]
        files2 = [os.path.join(path2, file) for file in files2]
        files3 = [os.path.join(path3, file) for file in files3]
        # files = files1 + files2 + files3
        files = files3

    elif dataset == "chaos_hd":
        path = "/home/luigi/Documents/QWT/datasets/chaos2019"
        files = ChaosDataset_Syn_new(path=path, split='train', modals=('t1', 't2', 'ct'), image_size=256)


    images = []
    counter = 0
    for file in tqdm(files):
        if dataset == "kvasir" or dataset == "chaos":
            if dataset == "kvasir":
                img = cv2.imread(os.path.join(path, file))
            elif dataset == "chaos":
                img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if dataset == "kvasir":
                img = cv2.resize(img, (512, 512))
        elif dataset == "ixi" or dataset == "kmnist" or dataset == "chaos_hd":
            img = file
        qwim = qw(img)
        images.append(qwim)
        counter += 1

    all_energies, subbands = step1(images)
    # subbands = [lglg, lghg, hglg, hghg, lflg, lfhg, hflg, hfhg, lglf, lghf, hglf, hghf, lflf, lfhf, hflf, hfhf]
    joint_pdfs = step2(list(subbands))

    counter_dictionary = repeat_kmeans(joint_pdfs=joint_pdfs, subbands=subbands, ntimes=kmeans_repetitions)
    print(counter_dictionary)

main("chaos_hd", kmeans_repetitions=20)


# Results best subbands per dataset:

### Pixels only ###
# Kvasir: Counter({0: 20, 4: 20, 13: 19, 7: 8, 10: 7, 11: 6})
# IXI: Counter({0: 20, 4: 20, 13: 20, 7: 9, 10: 7, 11: 4})
# KMNIST: Counter({0: 20, 13: 20, 3: 20, 2: 12, 7: 7, 4: 1})
# CHAOS: Counter({0: 20, 2: 20, 3: 19, 4: 18, 9: 2, 14: 1})
# CHAOS CT: Counter({0: 20, 2: 15, 4: 13, 3: 12, 9: 11, 13: 5, 8: 2, 14: 1, 7: 1})
# CHAOS T1: Counter({0: 20, 5: 20, 10: 19, 4: 17, 11: 3, 9: 1})


