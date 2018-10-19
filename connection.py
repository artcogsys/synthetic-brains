from chainer import Variable
from base import SBComponent
import numpy as np

#####
## SBConnection base class

class SBConnection(SBComponent):

    def __init__(self, n_out=1, delay=0):
        """ Implements a basic delay mechanism; default is no delay

        :param n_out: Number of outputs of the connection
        :param delay: Conduction delay in terms of number of sampling steps
        """

        super(SBConnection, self).__init__()

        self.n_out = n_out
        self._delay = delay
        self._history = None

    def __call__(self, x):
        """Forward propagation

        :param x: input to connection
        :return: connection output
        """

        if self._history is None:
            self._history = [Variable(np.zeros(x.shape, dtype='float32')) for i in range(self._delay)]

        self._history.append(x)
        y = self._history.pop(0)

        return y

    def reset(self):
        """Reset state
        """

        self._history = None