import torch
import torch.nn as nn
import numpy as np
from scipy.fftpack import hilbert as ht
from scipy.fftpack import ihilbert as iht
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from scipy.fftpack import hilbert as ht
import sys
import matplotlib.pyplot as plt
from scipy.fftpack import  ihilbert as iht
import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data


class QWTHelper():
    
    def __init__(self) -> None:
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bicubic')
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')

    
    def reflect(self, x, minx, maxx):
        """Reflect the values in matrix *x* about the scalar values *minx* and
        *maxx*.  Hence a vector *x* containing a long linearly increasing series is
        converted into a waveform which ramps linearly up and down between *minx* and
        *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers + 0.5), the
        ramps will have repeated max and min samples.
        .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
        .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.
        """
        # x = np.asanyarray(x)
        rng = maxx - minx
        rng_by_2 = 2 * rng
        mod = torch.fmod(x - minx, rng_by_2)
        normed_mod = torch.where(mod < 0, mod + rng_by_2, mod)
        out = torch.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
        return out

    def as_column_vector(self, v):
        """Return *v* as a column vector with shape (N,1).
        """
        v = torch.atleast_2d(v)
        if v.shape[0] == 1:
            return v.T
        else:
            return v

    def as_row_vector(self, v):
        """Return *v* as a column vector with shape (N,1).
        """
        v = torch.atleast_2d(v)
        if v.shape[0] == 1:
            return v
        else:
            return v.T

    def _centered(self, arr, newsize):
        # Return the center newsize portion of the array.
        # (Shamelessly cribbed from scipy.)
        currsize = torch.tensor(arr.shape)
        startind = torch.div(currsize-newsize, torch.tensor(2), rounding_mode='floor')
        #startind = (currsize - newsize) // 2
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
        
        h = h.flatten()
        h_size = h.shape[0]

        #     full_size = X.shape[0] + h_size - 1
        #     print("full size:", full_size)
        #     Xshape[0] = full_size

        out = torch.zeros_like(X)
        for idx in range(h_size):
            # conv = X*h[idx]
            out += X * h[idx]
        
        outShape = torch.tensor(X.shape)
        outShape[2] = abs(X.shape[2] - h_size) #+ 1

        return self._centered(out, outShape)

    def _row_convolve(self, X, h):
        """Convolve the columns of *X* with *h* returning only the 'valid' section,
        i.e. those values unaffected by zero padding. Irrespective of the ftype of
        *h*, the output will have the dtype of *X* appropriately expanded to a
        floating point type if necessary.
        We assume that h is small and so direct convolution is the most efficient.
        """
        
        h = h.flatten()
        h_size = h.shape[0]

        #     full_size = X.shape[0] + h_size - 1
        #     print("full size:", full_size)
        #     Xshape[0] = full_size

        out = torch.zeros_like(X)
        for idx in range(h_size):
            # conv = X*h[idx]
            out += X * h[idx]
        
        outShape = torch.tensor(X.shape)
        outShape[3] = abs(X.shape[3] - h_size) #+ 1

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
        # X = np.asarray(X)
        h = self.as_column_vector(h)
        batch, ch, r, c = X.shape
        m = h.shape[0]
        m2 = torch.fix(torch.tensor(m*0.5))

        # Symmetrically extend with repeat of end samples.
        # Use 'reflect' so r < m2 works OK.
        xe = self.reflect(torch.arange(-m2, r+m2), -0.5, r-0.5).long()
        # Perform filtering on the columns of the extended matrix X(xe,:), keeping
        # only the 'valid' output samples, so Y is the same size as X if m is odd.
        Y = self._column_convolve(X[:,:,xe], h)

        return Y


    def rowfilter(self, X, h):
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
        # X = np.asarray(X)
        h = self.as_row_vector(h)
        batch, ch, r, c = X.shape
        m = h.shape[1]
        m2 = torch.fix(torch.tensor(m*0.5))

        # Symmetrically extend with repeat of end samples.
        # Use 'reflect' so r < m2 works OK.
        xe = self.reflect(torch.arange(-m2, c+m2), -0.5, c-0.5).long()
        # Perform filtering on the columns of the extended matrix X(xe,:), keeping
        # only the 'valid' output samples, so Y is the same size as X if m is odd.
        Y = self._row_convolve(X[:,:,:,xe], h)

        return Y


class QWTForward(nn.Module):

    def __init__(self,device):
        super(QWTForward, self).__init__()
        self.device = device
        self.QWTHelper = QWTHelper()

        self.gl, self.gh, self.fl, self.fh = self.get_filters()
   
    def pywt_coeffs(self):
        # coefficients from pywt db8 (http://wavelets.pybytes.com/wavelet/db8/)
        gl = np.array([-0.00011747678400228192,
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
        ])

        gh = np.array([-0.05441584224308161,
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
        ])
        return gl, gh

    # Compute Hilbert transform of the filters G_L and G_H
    def get_hilbert_filters(self, gl, gh):
        fl = ht(gl)
        fh = ht(gh)
        return fl, fh

    def get_filters(self):
        gl, gh = self.pywt_coeffs()
        fl, fh = self.get_hilbert_filters(gl, gh)
        return torch.from_numpy(gl).to(self.device), \
            torch.from_numpy(gh).to(self.device),\
            torch.from_numpy(fl).to(self.device), \
            torch.from_numpy(fh).to(self.device)


    

    def forward(self, image):
        '''Compute the QWT. Just compute the low frequency coefficients.
        Return L_G L_G, L_F L_G, L_G L_F, L_F L_F.'''
    
        t1 = self.QWTHelper.colfilter(image, self.gl)
        t1 = self.QWTHelper.downsample(t1)
        lglg = self.QWTHelper.rowfilter(t1, self.gl)
        t2 = self.QWTHelper.colfilter(image, self.gl)
        t2 = self.QWTHelper.downsample(t2)
        lghg = self.QWTHelper.rowfilter(t2, self.gh)
        t3 = self.QWTHelper.colfilter(image, self.gh)
        t3 = self.QWTHelper.downsample(t3)
        hglg = self.QWTHelper.rowfilter(t3, self.gl)
        t4 = self.QWTHelper.colfilter(image, self.gh)
        t4 = self.QWTHelper.downsample(t4)
        hghg = self.QWTHelper.rowfilter(t4, self.gh)

        t1 = self.QWTHelper.colfilter(image, self.fl)
        t1 = self.QWTHelper.downsample(t1)
        lflg = self.QWTHelper.rowfilter(t1, self.gl)
        t2 = self.QWTHelper.colfilter(image, self.fl)
        t2 = self.QWTHelper.downsample(t2)
        lfhg = self.QWTHelper.rowfilter(t2, self.gh)
        t3 = self.QWTHelper.colfilter(image, self.fh)
        t3 = self.QWTHelper.downsample(t3)
        hflg = self.QWTHelper.rowfilter(t3, self.gl)
        t4 = self.QWTHelper.colfilter(image, self.fh)
        t4 = self.QWTHelper.downsample(t4)
        hfhg = self.QWTHelper.rowfilter(t4, self.gh)

        t1 = self.QWTHelper.colfilter(image, self.gl)
        t1 = self.QWTHelper.downsample(t1)
        lglf = self.QWTHelper.rowfilter(t1, self.fl)
        t2 = self.QWTHelper.colfilter(image, self.gl)
        t2 = self.QWTHelper.downsample(t2)
        lghf = self.QWTHelper.rowfilter(t2, self.fh)
        t3 = self.QWTHelper.colfilter(image, self.gh)
        t3 = self.QWTHelper.downsample(t3)
        hglf = self.QWTHelper.rowfilter(t3, self.fl)
        t4 = self.QWTHelper.colfilter(image, self.gh)
        t4 = self.QWTHelper.downsample(t4)
        hghf = self.QWTHelper.rowfilter(t4, self.fh)

        t1 = self.QWTHelper.colfilter(image, self.fl)
        t1 = self.QWTHelper.downsample(t1)
        lflf = self.QWTHelper.rowfilter(t1, self.fl)
        t2 = self.QWTHelper.colfilter(image, self.fl)
        t2 = self.QWTHelper.downsample(t2)
        lfhf = self.QWTHelper.rowfilter(t2, self.fh)
        t3 = self.QWTHelper.colfilter(image, self.fh)
        t3 = self.QWTHelper.downsample(t3)
        hflf = self.QWTHelper.rowfilter(t3, self.fl)
        t4 = self.QWTHelper.colfilter(image, self.fh)
        t4 = self.QWTHelper.downsample(t4)
        hfhf = self.QWTHelper.rowfilter(t4, self.fh)

        # return torch.cat((lglg, lghg, hglg, hghg, lflg, lfhg, hflg, hfhg, lglf, lghf, hglf, hghf, lflf, lfhf, hflf, hfhf),dim=1)[:,:,2:]
        return torch.cat((lglg, lflg, lglf, lflf), dim=1), [torch.stack([
                                                                            torch.cat((lghg, lfhg, lghf, lfhf),dim=1),
                                                                            torch.cat((hglg, hflg, hglf, hflf),dim=1), 
                                                                            torch.cat((hghg, hfhg, hghf, hfhf),dim=1)]
                                                                    ,dim=2)]




class QWTInverse(nn.Module):

    def __init__(self,device):
        super(QWTInverse, self).__init__()

        self.device = device
        self.QWTHelper = QWTHelper()
        self.gl, self.gh, self.fl, self.fh = self.get_filters()
   
    def pywt_coeffs(self):
        # coefficients from pywt db8 (http://wavelets.pybytes.com/wavelet/db8/)
        gl = np.array([0.05441584224308161,
                        0.3128715909144659,
                        0.6756307362980128,
                        0.5853546836548691,
                        -0.015829105256023893,
                        -0.2840155429624281,
                        0.00047248457399797254,
                        0.128747426620186,
                        -0.01736930100202211,
                        -0.04408825393106472,
                        0.013981027917015516,
                        0.008746094047015655,
                        -0.00487035299301066,
                        -0.0003917403729959771,
                        0.0006754494059985568,
                        -0.00011747678400228192
                        ])

        gh = np.array([-0.00011747678400228192,
                            -0.0006754494059985568,
                            -0.0003917403729959771,
                            0.00487035299301066,
                            0.008746094047015655,
                            -0.013981027917015516,
                            -0.04408825393106472,
                            0.01736930100202211,
                            0.128747426620186,
                            -0.00047248457399797254,
                            -0.2840155429624281,
                            0.015829105256023893,
                            0.5853546836548691,
                            -0.6756307362980128,
                            0.3128715909144659,
                            -0.05441584224308161,
                            ])
        return gl, gh

    # Compute Inverse Hilbert transform of the filters G_L and G_H
    def get_inverse_hilbert_filters(self, gl, gh):
        fl = iht(gl)
        fh = iht(gh)
        return fl, fh

    def get_filters(self):
        gl, gh = self.pywt_coeffs()
        fl, fh = self.get_inverse_hilbert_filters(gl, gh)
        return torch.from_numpy(gl).to(self.device), \
            torch.from_numpy(gh).to(self.device),\
            torch.from_numpy(fl).to(self.device), \
            torch.from_numpy(fh).to(self.device)
    
    

    def forward(self,tuple_with_wav):
        '''Compute the IQWT.'''
        (LL,Yh) = tuple_with_wav
        LH, HL, HH = Yh[0][:,:,0], Yh[0][:,:,1], Yh[0][:,:,2]
        split_size = LL.size(1) // 4

        lglg, lflg, lglf, lflf = LL.split(split_size=split_size,dim=1)
        lghg, lfhg, lghf, lfhf = LH.split(split_size=split_size,dim=1)
        hglg, hflg, hglf, hflf = HL.split(split_size=split_size,dim=1)
        hghg, hfhg, hghf, hfhf = HH.split(split_size=split_size,dim=1)

        
        lglg = self.QWTHelper.rowfilter(lglg, self.gl)
        t1 = self.QWTHelper.upsample(lglg)
        lghg = self.QWTHelper.rowfilter(lghg, self.gh)
        t2 = self.QWTHelper.upsample(lghg)
        
        t = t1+t2
        t1 = self.QWTHelper.colfilter(t, self.gl)
        
        hglg = self.QWTHelper.rowfilter(hglg, self.gl)
        t3 = self.QWTHelper.upsample(hglg)
        hghg = self.QWTHelper.rowfilter(hghg, self.gh)
        t4 = self.QWTHelper.upsample(hghg)

        t = t3+t4
        t2 = self.QWTHelper.colfilter(t, self.gh)

        first_component = t1 + t2
        ####
        lflg = self.QWTHelper.rowfilter(lflg, self.gl)
        t1 = self.QWTHelper.upsample(lflg)
        lfhg = self.QWTHelper.rowfilter(lfhg, self.gh)
        t2 = self.QWTHelper.upsample(lghg)
        
        t = t1+t2
        t1 = self.QWTHelper.colfilter(t, self.fl)
        
        hflg = self.QWTHelper.rowfilter(hflg, self.gl)
        t3 = self.QWTHelper.upsample(hflg)
        hfhg = self.QWTHelper.rowfilter(hfhg, self.gh)
        t4 = self.QWTHelper.upsample(hfhg)

        t = t3+t4
        t2 = self.QWTHelper.colfilter(t, self.fh)

        second_component = t1 + t2       
        ####
        lglf = self.QWTHelper.rowfilter(lglf, self.fl)
        t1 = self.QWTHelper.upsample(lglf)
        lghf = self.QWTHelper.rowfilter(lghf, self.fh)
        t2 = self.QWTHelper.upsample(lghf)
        
        t = t1+t2
        t1 = self.QWTHelper.colfilter(t, self.gl)
        
        hglf = self.QWTHelper.rowfilter(hglf, self.fl)
        t3 = self.QWTHelper.upsample(hglf)
        hghf = self.QWTHelper.rowfilter(hghf, self.fh)
        t4 = self.QWTHelper.upsample(hghf)

        t = t3+t4
        t2 = self.QWTHelper.colfilter(t, self.gh)

        third_component = t1 + t2       
        ####
        lflf = self.QWTHelper.rowfilter(lflf, self.fl)
        t1 = self.QWTHelper.upsample(lflf)
        lfhf = self.QWTHelper.rowfilter(lfhf, self.fh)
        t2 = self.QWTHelper.upsample(lfhf)
        
        t = t1+t2
        t1 = self.QWTHelper.colfilter(t, self.fl)
        
        hflf = self.QWTHelper.rowfilter(hflf, self.fl)
        t3 = self.QWTHelper.upsample(hflf)
        hfhf = self.QWTHelper.rowfilter(hfhf, self.fh)
        t4 = self.QWTHelper.upsample(hfhf)

        t = t3+t4
        t2 = self.QWTHelper.colfilter(t, self.fh)

        fourth_component = t1 + t2       
        ####
        y = first_component + second_component + third_component + fourth_component
        y = (y-y.min()) / (y.max() - y.min())
        return y





def main():
    from PIL import Image
    img_original = pywt.data.camera()
    img = torch.from_numpy(img_original).unsqueeze(0).unsqueeze(0).float() / 255.0

    qwt = QWTForward("cpu")
    iqwt = QWTInverse("cpu")

    wavelets = qwt(img)
    LL,Yh = wavelets
    #to test
    iqwt_out = iqwt(wavelets)
    print((iqwt_out - img))
    print(torch.max(img),torch.mean(img), torch.min(img))
    print(torch.max(iqwt_out),torch.mean(iqwt_out), torch.min(iqwt_out))

    print(LL.shape)
    print(Yh[0].shape)
    print(iqwt_out.shape)
    
    out_img_plot = Yh[0][:,0,].squeeze(0).detach().cpu().numpy()
    iqwt_out_plot = iqwt_out[0].squeeze(0).detach().cpu().numpy()
    
    print(np.max(out_img_plot),np.mean(out_img_plot), np.min(out_img_plot))

    plt.subplot(1,3,1)
    plt.imshow(img_original, cmap=plt.cm.gray)
    plt.subplot(1,3,2)
    plt.imshow(out_img_plot[0], cmap=plt.cm.gray)
    plt.subplot(1,3,3)
    plt.imshow(iqwt_out_plot, cmap=plt.cm.gray)
    #plt.show()
    plt.savefig("test.png")


# main()