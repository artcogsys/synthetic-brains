import chainer.links as L
import chainer.functions as F
from chainer import Variable
from base import SBComponent
import numpy as np
import scipy.stats as sps

#####
## SBReadout base class

# Readouts transform lists of population responses to a chainer variable array

class SBReadout(SBComponent):

    # Custom readout mechanism just copies the population response
    # It assumes that population activity can be read out directly

    def __call__(self, x):

        return F.concat(x)


class CanonicalHRFReadout(SBComponent):

    def __init__(self, delta_t):
        """

        Implements a canonical HRF. Stores a history of past responses and transforms this using a canonical HRF
        into an observed BOLD response

        :param delta_t: time in seconds between time slices
        """

        super(SBComponent, self).__init__()

        self._T_MAX = 12  # number of seconds to integrate over

        self.delta_t = delta_t  # delta t

        self._n_steps = np.ceil(self._T_MAX / self.delta_t).astype('int32')  # nr of time slices to consider

        self.hrf = spm_hrf_compat(self.delta_t * np.arange(self._n_steps)) # hrf weightings

        self._history = None

    def __call__(self, x):
        """Forward propagation

        :param x: input to connection
        :return: connection output
        """

        response = F.concat(x)

        if self._history is None:
            self._history = [Variable(np.zeros(response.shape, dtype='float32')) for ix in range(self._n_steps)]

        self._history.append(response)
        self._history.pop(0)

        bold = Variable(np.zeros(response.shape, 'float32'))
        for i in range(len(self._history)):
            bold += self.hrf[-i] * self._history[i]

        # multiply population responses with hrf and sum
        # bold = F.sum(F.concat(self._history, axis=0) * np.repeat(np.atleast_2d(self.hrf[::-1]), 2, axis=0).T, axis=0)

        return bold

    def reset(self):
        """Reset state
        """

        self._history = None



def spm_hrf_compat(t,
                   peak_delay=6,
                   under_delay=16,
                   peak_disp=1,
                   under_disp=1,
                   p_u_ratio = 6,
                   normalize=True,
                  ):
    """

    From NiPy

    SPM HRF function from sum of two gamma PDFs
    This function is designed to be partially compatible with SPMs `spm_hrf.m`
    function.
    The SPN HRF is a *peak* gamma PDF (with location `peak_delay` and dispersion
    `peak_disp`), minus an *undershoot* gamma PDF (with location `under_delay`
    and dispersion `under_disp`, and divided by the `p_u_ratio`).
    Parameters
    ----------
    t : array-like
        vector of times at which to sample HRF.
    peak_delay : float, optional
        delay of peak.
    under_delay : float, optional
        delay of undershoot.
    peak_disp : float, optional
        width (dispersion) of peak.
    under_disp : float, optional
        width (dispersion) of undershoot.
    p_u_ratio : float, optional
        peak to undershoot ratio.  Undershoot divided by this value before
        subtracting from peak.
    normalize : {True, False}, optional
        If True, divide HRF values by their sum before returning.  SPM does this
        by default.
    Returns
    -------
    hrf : array
        vector length ``len(t)`` of samples from HRF at times `t`.
    Notes
    -----
    See ``spm_hrf.m`` in the SPM distribution.
    """
    if len([v for v in [peak_delay, peak_disp, under_delay, under_disp]
            if v <= 0]):
        raise ValueError("delays and dispersions must be > 0")
    # gamma.pdf only defined for t > 0
    hrf = np.zeros(t.shape, dtype=np.float)
    pos_t = t[t > 0]
    peak = sps.gamma.pdf(pos_t,
                         peak_delay / peak_disp,
                         loc=0,
                         scale = peak_disp)
    undershoot = sps.gamma.pdf(pos_t,
                               under_delay / under_disp,
                               loc=0,
                               scale = under_disp)
    hrf[t > 0] = peak - undershoot / p_u_ratio
    if not normalize:
        return hrf
    return hrf / np.sum(hrf)

# #####
# ## Linear readout
#
# class DRMReadout2(SBComponent):
#
#     def __init__(self, n_out=1):
#
#         super(DRMReadout2, self).__init__()
#
#         self.n_out = n_out
#
#         self.l1 = L.Linear(None, n_out)
#
#     def __call__(self, x):
#         """Forward propagation
#
#         :param x: readout input
#         :type x: list of afferent population outputs
#         :return: predicted measurements
#         """
#
#         # this readout mechanism concatenates all population outputs for further processing
#         x = F.concat(x, axis=1)
#
#         return self.l1(x)
