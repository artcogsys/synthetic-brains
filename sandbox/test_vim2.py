#######
# Test basic model on VIM2 data

from iterators import SBIterator
from population import NPPopulation
from connection import SBConnection
from base import SBEstimator, SyntheticBrain
from readout import DRMReadout2
import numpy as np
import matplotlib.pyplot as plt
import tables
from scipy.stats import zscore
import pickle
import Image

# sampling resolution in seconds converted from stimulation frequency (15 Hz)
resolution = 1.0/15

# cutoff in seconds converted to number of time steps
cutoff =  np.ceil(30 / resolution)

# number of training epochs
n_epochs = 5 # 100

#######
# Preprocessing

if False:

    # load training and test data for the first subject in the training data set
    # NOTE: we may want to use the non-averaged validation data

    # read stimuli
    f = tables.openFile('/Users/marcelvangerven/Data/VIM-2/Stimuli.mat')
    train_stimuli = f.get_node('/st')[:] # shape: (108000, 3, 128, 128)
    test_stimuli = f.get_node('/sv')[:] # (8100, 3, 128, 128)
    f.close()

    # convert stimuli to greyscale and resize
    train_stimuli = np.asarray([np.asarray(Image.fromarray(np.rollaxis(train_stimuli[i,:,:,:],0,3), 'RGB').convert('L').resize([32, 32])) for i in range(train_stimuli.shape[0]) ])
    test_stimuli = np.asarray([np.asarray(Image.fromarray(np.rollaxis(test_stimuli[i,:,:,:],0,3), 'RGB').convert('L').resize([32, 32])) for i in range(test_stimuli.shape[0]) ])

    # flatten data
    train_stimuli = train_stimuli.reshape(train_stimuli.shape[0],-1)
    test_stimuli = test_stimuli.reshape(test_stimuli.shape[0],-1)

    # zscore data
    train_stimuli = zscore(train_stimuli)
    test_stimuli = zscore(test_stimuli)

    # read responses
    f = tables.openFile('/Users/marcelvangerven/Data/VIM-2/VoxelResponses_subject1.mat')
    train_response = f.get_node('/rt')[:]
    test_response = f.get_node('/rv')[:]

    # select left hemisphere V1 voxel responses only
    roi = f.get_node('/roi/v1lh')[:].flatten()
    v1lh_idx = np.nonzero(roi==1)[0]
    f.close()

    train_response = train_response[v1lh_idx].T # shape: (7200, 494)
    test_response = test_response[v1lh_idx].T # shape: (540, 494)

    # take out voxels that have no data (nan)
    nnan_idx = np.where(~np.any(np.isnan(train_response), axis=0))
    train_response = np.squeeze(train_response[:,nnan_idx])
    test_response = np.squeeze(test_response[:,nnan_idx])

    # debugging - reduce data to something managable
    train_stimuli = np.atleast_2d(np.mean(train_stimuli,1)).T[:15000,:] # mean luminance signal for first 1000 seconds
    test_stimuli = np.atleast_2d(np.mean(test_stimuli,1)).T[:1500,:] # mean luminance signal for first 100 seconds
    train_response = train_response[:1000,:2] # first two voxels for first 1000 seconds
    test_response = test_response[:100,:2] # first two voxels for first 100 seconds

    # store data
    f = open('/Users/marcelvangerven/Code/github/SBEstimator/store.pckl', 'wb')
    pickle.dump([train_stimuli, train_response, test_stimuli, test_response], f)
    f.close()

else:

    # load data
    f = open('/Users/marcelvangerven/Code/github/SBEstimator/store.pckl', 'rb')
    train_stimuli, train_response, test_stimuli, test_response = pickle.load(f)
    f.close()

#######
# Create times at which stimuli and responses occur

train_stim_time = [i for i in range(train_stimuli.shape[0])]
test_stim_time = [i for i in range(test_stimuli.shape[0])]

train_resp_time = [i*15 for i in range(train_response.shape[0])] # TR of 1 second; so one response every 15 time steps
test_resp_time = [i*15 for i in range(test_response.shape[0])]  # TR of 1 second; so one response every 15 time steps

#######
# Create iterators which generate stimuli and responses

train_iter = SBIterator(stimulus=train_stimuli, stim_time=train_stim_time,
                        response=train_response, resp_time=train_resp_time, batch_size=32)

test_iter = SBIterator(stimulus=test_stimuli, stim_time=test_stim_time,
                       response=test_response, resp_time=test_resp_time, batch_size=32)

###
# CREATE MODEL - one population only

n_voxels = train_response.shape[1] # number of outputs

# standard populations
populations = [NPPopulation()]

# linear readout
readout = DRMReadout2(n_out=n_voxels)

# link stimulus to all populations; each population receives the (delayed) stimulus input
ws = [SBConnection()]

# create population matrix - no connections
Wp = np.array([None]).reshape([1, 1])

# set up ground truth model - run on cpu or gpu
drm_model = SyntheticBrain(populations, ws, Wp, readout, gpu=-1)

drm = SBEstimator(drm_model)

#######
# compute validation loss on initial model
initial_loss = drm.compute_loss(test_iter)

#######
# estimate model

# run SBEstimator
train_loss, validation_loss = drm.estimate(train_iter, val_iter=test_iter, n_epochs=n_epochs, cutoff=cutoff)

#######
# check decrease in loss

print initial_loss

plt.figure()
plt.plot(train_loss)
plt.hold(True)
plt.plot(validation_loss)
plt.legend(['training loss', 'validation loss'])
plt.show()

