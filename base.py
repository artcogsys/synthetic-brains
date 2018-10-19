import chainer
from chainer import ChainList, Chain
from chainer import Variable
from chainer import cuda
import chainer.functions as F
import numpy as np
import tqdm
import copy

class SBComponent(Chain):
    """Synthetic Brain Component
       Base class for populations, readouts and connections
    """

    def reset(self):
        """ The function that is called when resetting internal state
        """

        pass


class SyntheticBrain(ChainList):
    """SyntheticBrain; the recurrent neural network which includes populations, connections and readouts
    """

    def __init__(self, populations, ws, Wp, readout, gpu=-1):
        """ SyntheticBrain initializer

        Each population receives either sensory input or input from other populations.
        This reception is mediated by connections which are neural networks themselves.
        There are three kinds of connections: sensory-population, population-population, population-response
        These all derive from the same object but we can have specific default implementations. Absent connections are
        represented as None.

        :param populations: npop list specifying each population
        :param ws: list specifying a connection between the stimulus and each population.
        :param Wp: npop x npop object array specifying a connection between all populations
        :param readout: readout mechanism
        :param gpu: run on cpu or gpu
        """

        super(SyntheticBrain, self).__init__()

        self.n_pop = len(populations)

        # add component links

        for i in range(self.n_pop):
            self.add_link(populations[i])

        self.add_link(readout)

        for i in range(self.n_pop):
            if not ws[i] is None:
                self.add_link(ws[i])

        wp = np.ravel(Wp)
        for i in range(Wp.size):
            if not wp[i] is None:
                self.add_link(wp[i])

        # add populations
        self.populations = populations

        # add connections

        self.ws = ws

        self.Wp = Wp

        # add readout mechanism
        self.readout = readout

        # store population activity
        self.pop_output = []

        # cpu/gpu mode
        if gpu >= 0:
            cuda.get_device(gpu).use()
            self.to_gpu()


    def __call__(self, data):
        """Forward propagation for one step

        :param data: sensory input at this point in time (zeros for no input); numpy array
        :return: predicted output measurements
        """

        # optionally move to gpu
        x = Variable(self.xp.asarray(data))

        batch_size = x.shape[0]

        # initialize population outputs
        self.pop_output = [Variable(self.xp.zeros([batch_size, p.n_out], dtype='float32')) for p in self.populations]

        # randomly update each population
        for i in np.random.permutation(self.n_pop):

            # pass sensory input to the sensory-population connection associated with this population
            # the result is the sensory input for that population (e.g. delayed input)
            ws = self.ws[i]
            if ws is None:
                pop_input = []
            else:
                # push stimulus into connection that links stimulus to i-th population
                pop_input = [self.ws[i](x)]

            # get population connections entering this population and pass other populations output through the connections
            # note that population outputs can be zero or not depending on whether they have been updated already
            wp = self.Wp[i]
            for j in range(self.n_pop):
                if not wp[j] is None:
                    # push output of j-th population into connection that links it to i-th population
                    pop_input.append(wp[j](self.pop_output[j]))

            # pop_input now contains the output of all connections that provide the input to the i-th population

            # compute population output for the i-th population
            self.pop_output[i] = self.populations[i](pop_input)

        # now we have all the outputs, we can pass it to the readout mechanism
        return self.readout(self.pop_output)

    def loss(self, stimulus, response):
        """Computes loss on a prediction from stimulus and a target response

        Computes MSE loss but ignores those terms where the target is equal to nan, indicating missing data.

        :param stimulus: input stimulus
        :param response: Target output
        :type prediction: Variable
        :type target: Variable
        :return: MSE loss
        :rtype: Variable
        """

        prediction = self(stimulus)

        target = Variable(self.xp.asarray(response))

        idx = np.where(np.any(np.isnan(target.data), axis=1) == False)[0].tolist()

        if idx:
            return F.mean_squared_error(prediction[idx,:], target[idx,:])
        else:
            return Variable(self.xp.zeros((), 'float32'))

    def reset(self):
        """ Reset states of model components
        """

        for i in range(self.n_pop):

            # reset stimulus connections
            if not self.ws[i] is None:
                self.ws[i].reset()

            # reset population connections
            wp = self.Wp[i]
            for j in range(self.n_pop):
                if not wp[j] is None:
                    wp[j].reset()

            self.populations[i].reset()

        self.readout.reset()

    def run(self, data_iter):
        """Forward propagation of synthetic brain on a data iterator

        :param data_iter:
        :return: generated response and population activity
        """

        response = []
        activity = []

        self.reset()

        with chainer.using_config('train', False):

            for data in data_iter:

                # r = [x.data[0, 0] for x in self(data['stimulus'])]
                r = self(data['stimulus']).data

                # keep track of population activity
                activity.append(np.array([x.data[0, 0] for x in self.pop_output]))

                response.append(r)

        return np.vstack(response), np.vstack(activity)


class SBEstimator(object):
    """wrapper object that trains and analyses the model at hand
    """

    def __init__(self, sb):
        """

        :param sb: a synthetic brain
        """

        self.model = sb

        # stores optimal model according to validation loss
        self._optimal_model = copy.deepcopy(self.model)

        # optimizer
        self.optimizer = chainer.optimizers.Adam()
        self.optimizer.setup(self.model)

    def fit(self, data_iter, val_iter=None, n_epochs=1, cutoff=None):
        """Model estimation via truncated backprop

        :param data_iter: iterator which generates sensations/responses at some specified resolution
        :param val_iter: optional iterator which generates sensations/responses at some specified resolution used for validation
        :param n_epochs: number of training epochs
        :param cutoff: cutoff for truncated backpropagation in terms of number of time slices
        :return: train loss and validation loss
        """

        # initialization for validation
        min_loss = None

        # track training and validation loss
        train_loss = np.zeros(n_epochs)
        validation_loss = np.zeros(n_epochs)

        idx = 0
        for epoch in tqdm.trange(n_epochs):

            # keep track of loss
            loss = None

            # reset at start of each epoch
            self.model.reset()

            with chainer.using_config('train', True):

                for data in data_iter:

                    # compute training loss
                    _loss = self.model.loss(data['stimulus'], data['response'])

                    if loss is None:
                        loss = _loss
                    else:
                        loss += _loss

                    train_loss[epoch] += _loss.data

                    # backpropagate if we reach the cutoff for truncated backprop or if we processed the last batch
                    if (cutoff and idx == cutoff-1) or data_iter.is_final():

                        self.model.cleargrads()
                        _loss.backward()
                        _loss.unchain_backward()
                        self.optimizer.update()

                        loss = None

                        idx = 0

                    idx += 1

            # run validation
            if not val_iter is None:

                # compute validation loss
                validation_loss[epoch] = self.compute_loss(val_iter)

                # store best model in case loss was minimized
                if min_loss is None or validation_loss[epoch] < min_loss:
                    self._optimal_model = copy.deepcopy(self.model)
                    min_loss = validation_loss[epoch]

        # set model to optimal model
        if not val_iter is None:
            self.model = copy.deepcopy(self._optimal_model)

        return train_loss, validation_loss

    def compute_loss(self, data_iter):
        """Compute loss on data iterator in test mode
        
        :param data_iter: 
        :return: loss
        """

        loss = 0

        # reset at start of each epoch
        self.model.reset()

        with chainer.using_config('train', False):

            for data in data_iter:

                # compute training loss
                loss += self.model.loss(data['stimulus'], data['response']).data

        return loss
