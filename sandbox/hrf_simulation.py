# Demonstrates recovery of model parameters when the readout is a (simulated) BOLD response

import numpy as np
import matplotlib.pyplot as plt

from iterators import SBIterator
from population import NPPopulation
from connection import SBConnection
from base import SyntheticBrain, SBEstimator

# Canonical HRF
from readout import CanonicalHRFReadout

# Define parameters

#######
# Model parameters

n_stim = 5  # number of input stimuli
n_pop = 1   # number of assumed neural populations
n_resp = 1  # number of measured responses

delta_t = 1  # time between time slices in seconds (used to transform event times to real time)

stim_delay = 0  # delay between input stimulus and populations (in number of time slices)
pop_delay = 0   # delay between populations (in number of time slices)

#######
# Inference Parameters

n_epochs = 1000  # number of epochs
cutoff = 5  # cutoff for truncated backpropagation

#######
# Data parameters

stim_len = 500  # number of stimuli presented
resp_len = 500  # number of measured responses

stim_res = 1  # stimulus resolution (in number of time slices)
resp_res = 1  # response measurement resolution (in number of time slices)

stim_offset = 0  # stimulus offset relative to start of population sampling
resp_offset = 0  # response offset relative to start of population sampling

# actual time slices at which stimuli/responses are sampled
stim_time = np.arange(stim_offset, stim_offset + stim_len * stim_res, stim_res)
resp_time = np.arange(resp_offset, resp_offset + resp_len * resp_res, resp_res)

# Define helper function. Creates an NPModel with a particular structure. Used to create multiple instances with
# different initial parameters. The default model use populations with a linear layer, connections that have zero
# delay, and readouts that directly observe the population outputs.

def create_model():
    """

    :return: NPModel object
    """

    # standard populations
    populations = [NPPopulation() for ix in range(n_pop)]

    readout = CanonicalHRFReadout(delta_t=delta_t)

    # link stimulus to all populations; each population receives the (delayed) stimulus input
    ws = [SBConnection(delay=stim_delay) for ix in range(n_pop)]

    # create full population matrix (i.e. all populations connect to one another
    Wp = np.array(n_pop ** 2 * [None]).reshape([n_pop, n_pop])
    for i in range(n_pop):
        for j in range(n_pop):
            if i != j:
                Wp[i, j] = SBConnection(delay=pop_delay)

    # set up ground truth model
    model = SyntheticBrain(populations, ws, Wp, readout)

    return model


gt_model = create_model()


#######
# Generate responses based on sensory input when running the ground truth model in forward mode

# data used for model estimation
stimulus1 = np.random.randn(stim_len, n_stim)
data_iter = SBIterator(stimulus=stimulus1, stim_time=stim_time, batch_size=1)
response1, activity1 = gt_model.run(data_iter)

# We only keep responses that are associated with the response_times
response1 = response1[resp_time]

# data used for model validation
stimulus2 = np.random.randn(stim_len, n_stim)
data_iter = SBIterator(stimulus=stimulus2, stim_time=stim_time, batch_size=1)
response2, activity2 = gt_model.run(data_iter)

# We only keep responses that are associated with the response_times
response2 = response2[resp_time]

# Redefine iterators such that they generate both stimuli and responses
data_iter = SBIterator(stimulus=stimulus1, stim_time=stim_time, response=response1, resp_time=resp_time, batch_size=32)
val_iter = SBIterator(stimulus=stimulus2, stim_time=stim_time, response=response2, resp_time=resp_time, batch_size=32)

# plot some of the generated training data

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(delta_t * stim_time, stimulus1)
plt.title('Input stimulus')
plt.xlabel('time (s)')
plt.subplot(3, 1, 2)
plt.plot(delta_t * np.arange(activity1.shape[0]), activity1)
plt.title('Population activity')
plt.xlabel('time (s)')
plt.subplot(3, 1, 3)
plt.plot(delta_t * resp_time, response1)
plt.title('Measured response')
plt.xlabel('time (s)')
plt.show(block=False)

# Create model with same structure as ground truth model but different initial parameters

model = create_model()

# Create data for model testing

test_stim = np.random.randn(stim_len, n_stim)
test_iter = SBIterator(stimulus=test_stim, stim_time=stim_time, batch_size=1)

# Compute measured responses and population activity for real model and untrained initial model
response_gt, activity_gt = gt_model.run(test_iter)
response_init, activity_init = model.run(test_iter)


#######
# estimate model

est = SBEstimator(model)

train_loss, validation_loss = est.fit(data_iter, val_iter, n_epochs, cutoff=cutoff)

#######
# plot change in loss

plt.figure()
plt.plot(train_loss)
plt.plot(validation_loss)
plt.legend(['training loss', 'validation loss'])
plt.show(block=False)


#######
# compute MSE between population responses for initial and estimated model wrt ground truth model

# compute responses and population activity for estimated model

response_estim, activity_estim = model.run(test_iter)

# compute MSE between population activity of real and initial model
c1 = []
for i in range(n_pop):
    c1.append(((np.squeeze(activity_gt[:,i]) - np.squeeze(activity_init[:,i])) ** 2).mean(axis=0))

# compute MSE between population activity of real and estimated model
c2 = []
for i in range(n_pop):
    c2.append(((np.squeeze(activity_gt[:,i]) - np.squeeze(activity_estim[:,i])) ** 2).mean(axis=0))


print('MSE for initial model:')
print(c1)
print('MSE for estimated model:')
print(c2)

#######
# plot population activity - first 100 datapoints

plt.figure()
for i in range(n_pop):

    x = np.vstack([activity_gt[:, i], activity_init[:, i], activity_estim[:, i]]).T

    plt.subplot(n_pop,1,i+1)
    plt.plot(x[:100])
    plt.title('population {0}; initial MSE={1:4.3f}; estim MSE={2:4.3f}'.format(i, c1[i], c2[i]))
    plt.legend(['ground truth', 'initial', 'estimate'])

plt.show(block=False)

#######
# plot population activity - scatterplot between gt and estimated population activity

plt.figure()
for i in range(n_pop):

    x = np.vstack([activity_gt[:, i], activity_init[:, i], activity_estim[:, i]]).T

    plt.subplot(n_pop,1,i+1)
    plt.scatter(activity_gt[:, i], activity_init[:, i], c='r', label='initial')
    plt.scatter(activity_gt[:, i], activity_estim[:, i], c='b', label='fitted')
    plt.xlabel('true population activity')
    plt.ylabel('estimated population activity')


plt.show()
