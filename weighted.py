# Author: Eric Bezzam
# Date: July 15, 2016
"""
Modified by Kuan-Lin Chen on Sat Mar 5 2022
"""
import numpy as np

from pyroomacoustics.doa.music import *


class Weighted(MUSIC):

    def __init__(
        self,
        L,
        fs,
        nfft,
        c=343.0,
        num_src=1,
        mode="far",
        r=None,
        azimuth=None,
        colatitude=None,
        frequency_normalization=False,
        **kwargs
    ):

        super().__init__(
            L,
            fs,
            nfft,
            c,
            num_src,
            mode,
            r,
            azimuth,
            colatitude,
            frequency_normalization,
            **kwargs
        )

    def _process(self, X):
        self.Pssl = np.zeros((self.num_freq, self.grid.n_points))
        C_hat = self._compute_correlation_matricesvec(X)
        self.Pssl = self._compute_spatial_spectrumvec(C_hat)
        if self.frequency_normalization:
            self._apply_frequency_normalization()
        self.grid.set_values(np.squeeze(np.sum(self.Pssl, axis=1) / self.num_freq))

    def _compute_spatial_spectrumvec(self, cross):
        mod_vec = np.transpose(
            np.array(self.mode_vec[self.freq_bins, :, :]), axes=[2, 0, 1]
        )

        denom = np.matmul(
            np.conjugate(mod_vec[..., None, :]), np.matmul(cross, mod_vec[..., None])
        )
        return abs(denom[..., 0, 0])

    def _compute_spatial_spectrum(self, cross, k):

        P = np.zeros(self.grid.n_points)

        for n in range(self.grid.n_points):
            Dc = np.array(self.mode_vec[k, :, n], ndmin=2).T
            Dc_H = np.conjugate(np.array(self.mode_vec[k, :, n], ndmin=2))
            denom = np.dot(np.dot(Dc_H, cross), Dc)
            P[n] = abs(denom)

        return P
