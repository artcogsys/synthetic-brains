# Synthetic brains

We develop differentiable synthetic brains (SBs) that can be estimated from data. 
SBs fully consist of differentiable components (typically simple recurrent neural networks) that can be estimated end-to-end using stochastic gradient 
descent. SBs are composed of:

- populations: Models the responses of neural populations
- readouts: Models the readout of neural activity or behavioural responses
- connections: Models the connections between external inputs (sensations) and populations as well as between neural populations.

## Connections

Models the connections between a sensory input and neural populations as well as between neural populations. Connections 
between external inputs and populations can be instantaneous. Connections between populations must have some delay since 
the full model unfolds into a recurrent architecture.


## Readout

The readout mechanism operates on all populations at once. I.e. the readout mechanism is responsible for very specific 
readouts of particular populations. The default readout is just a direct readout of the populations. Readout mechanisms 
may also have adjustable parameters.

## Iterator

The Iterator object is responsible for the generation of stimuli and/or responses. The data stream is sampled at a 
particular sampling rate. At each point in time, stimuli and/or responses can be either present or absent depending 
on the stim_times and resp_times. This leads to a partially observed stream both on the input and output side. The 
stream can generate data in batch mode. Only generates stimulus input in case response=None which can be used for 
forward simulation. 

The sampling resolution is determined by the update of the populations (each timeslice). Stimulus times and response 
times are defined relative to this update frequency (as integers). Hence, care must be taken when converting from and to
real units of time.