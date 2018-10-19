# TO DO

* The model should be run until the last stimulus or response is processed
* The sampling times should be indicated as integer multiples of the sampling resolution.
  This makes things easier and prevents roundoff problems.
* Generalize stimuli to their onsets and offsets
* How to optimally deal with small dataset (i.e. truncation; batch selection, etc)
* Test on (V1/V2) BOLD data (Michele); two populations; conditioned on V1 or V2 voxels only
* Test on Allen/zebrafish data (Tami)
* Test for recovery of ME model on FMRI and CRCNS monkey data
* Link model to physical constants; i.e. make link to biophysics explicit
* Create learnable delays
* Improve model plausibility
* Make model work on resting state data (using stochastic nodes and other loss functions)
* Make model suitable for direct training on tasks
* Add visualizations; inspection of internal states; graphical ways of building the model
* Improve and expand documentation using Sphinx: http://www.sphinx-doc.org/en/stable/
* Develop readout mechanisms for fMRI/MEG/Calcium/LFP/Spiking data
* Hogwild style update. Link to biophysical, Euler and other updates
* SBEstimator will be the general framework for biocircuit informed general modeling
* GRU motivation; see Miconi paper
* Identifiability issues: how to determine if a model does well when comparing against ground truth data? Parameter identifiability but also population identifiability in case of models that contain exactly the same populations
* Add deepcopy functionality
* Create multiple readout mechanisms? No because we cannot handle e.g. MEG data
* Add unit testing