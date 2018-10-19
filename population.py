import chainer.links as L
import chainer.functions as F
from base import SBComponent

#####
## NPPopulation classes

class NPPopulation(SBComponent):
    """Default population is a linear layer with ReLU output
    """

    def __init__(self, n_out=1):

        super(NPPopulation, self).__init__()

        self.n_out = n_out

        with self.init_scope():
            self.l1 = L.Linear(None, n_out)

    def __call__(self, x):
        """Forward propagation

        :param x: population input
        :type x: list of afferent connection outputs
        :return: population output
        """

        # the list of inputs (e.g. stimulus and other populations) are concatenated for further processing
        x = F.concat(x, axis=1)

        # NOTE: relu unstable when we have few units and the input is always negative (zero gradient)
        # In that case leaky_relu may be better (though semantics of negative firing rates ...)
        return F.relu(self.l1(x))


class NPGRUPopulation(SBComponent):
    """Population with temporal dynamics using a GRU unit
    """

    def __init__(self, n_out=1):

        super(NPGRUPopulation, self).__init__()

        self.n_out = n_out

        with self.init_scope():
            self.l1 = L.GRU(None, n_out)

    def __call__(self, x):
        """Forward propagation

        :param x: population input
        :type x: list of afferent connection outputs
        :return: population output
        """

        # the list of inputs (e.g. stimulus and other populations) are concatenated for further processing
        x = F.concat(x, axis=1)

        # NOTE: relu unstable when we have few units and the input is always negative (zero gradient)
        # In that case leaky_relu may be better (though semantics of negative firing rates ...)
        return F.relu(self.l1(x))

    def reset(self):
        """
        Reset state. Required for GRU at onset of each training epoch
        """

        self.l1.reset_state()