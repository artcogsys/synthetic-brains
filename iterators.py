import numpy as np

#####
## SB iterator

class SBIterator(object):

    def __init__(self, stimulus, stim_time, response=None, resp_time=None, batch_size=None, n_batches=None):
        """ Initializer

        Generates stimulus and response outputs. The data stream is sampled at a particular sampling rate. At each
        point in time, stimuli and/or responses can be either present or absent depending on the stim_times and resp_times.
        This leads to a partially observed stream both on the input and output side. The stream can generate data in batch
        mode. Only generates stimulus input in case response=None which can be used for forward simulation.

        :param stimulus: input stimulus - nsamples x d1 x ... numpy array (float32)
        :param stim_time: integer array of times at which stimuli were presented relative to start of simulation. 
                          Times are integers referring to the time slices, starting at t=0
        :param response: output responses - nsamples x d1 x ... numpy array (float32)
        :param resp_time: integer array of times at which responses were observed relative to start of simulation.
                          Times are integers referring to the time slices, starting at t=0
        :param batch_size: number of batches to process sequentially
        :param n_batches: number of time steps to take per batch
        """

        # set stimulus
        self.stimulus = stimulus
        self.n_in = self.stimulus[0].size

        # times at which stimuli were presented
        self.stim_time = stim_time

        # check if lengths agree
        assert(len(self.stimulus) == len(self.stim_time))

        self.response = response

        if self.response is not None:

            self.n_out = self.response[0].size

            self.resp_time = resp_time

            assert(len(self.response) == len(self.resp_time))

            # determine total number of time steps to take according to temporal resolution
            self.n_steps = (np.max([np.max(self.stim_time), np.max(self.resp_time)])+1).astype('int32')

        else:

            self.n_out = None
            self.resp_time = None

            # determine total number of time steps to take according to temporal resolution
            self.n_steps = (np.max(self.stim_time)+1).astype('int32')

        # by default we run once through the whole dataset
        if batch_size is None:
            batch_size = 1

        # division into number of batches (time steps in terms of population updates)
        if n_batches is None:
            n_batches = self.n_steps // batch_size

        self.batch_size = batch_size
        self.n_batches = n_batches

    def __iter__(self):
        """ Initializes data generator. Should be invoked at the start of each epoch

        :return: self
        """

        self.idx = 0

        # select batch indices at which to start sampling
        offsets = [i * self.n_batches for i in range(self.batch_size)]

        # determine time points to sample
        self._order = []
        for iter in range(self.n_batches):
            x = [(offset + iter) % self.n_steps for offset in offsets]
            self._order += x

        return self

    def __next__(self):
        """Produces next data item

        :return: dictionary containing the stimulus and the response
        """

        if self.idx == self.n_batches:
            raise StopIteration

        # recover time steps for this batch
        i = self.idx * self.batch_size
        sample_times = self._order[i:(i + self.batch_size)]

        # find for sample time if there is a corresponding stimulus
        idx = map(lambda t: np.where(self.stim_time == t)[0], sample_times)

        # create partially observed data (zeros for no input)
        stim_data = np.array(list(map(lambda x: np.zeros(self.stimulus[0].shape) if len(x) == 0 else self.stimulus[x[0]], idx))).astype('float32')

        data = {}
        data['stimulus'] = stim_data

        if self.response is not None:

            # find for sample time if there is a corresponding response
            idx = map(lambda t: np.where(self.resp_time == t)[0], sample_times)

            # create partially observed data (nans for no output)
            resp_data = np.array(list(map(lambda x: np.full(self.response[0].shape, np.nan) if len(x) == 0 else self.response[x[0]], idx))).astype('float32')

            data['response'] = resp_data

        self.idx += 1

        return data

    def is_final(self):
        """Flags if final iteration is reached

        :return: boolean if final batch is reached
        """

        return (self.idx==self.n_batches)