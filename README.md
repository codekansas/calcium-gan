# calcium-gan

Tutorial on building a GAN model to calcium imaging.

## Abstract

In neuroscience, one of the most painful tasks is obtaining reliable recordings of neuron spiking behavior. A popular idea approach is to use calcium imaging is to show the spiking behavior of a group of neurons without having to put electrodes in each individual cell. This uses a calcium indicator dye, which fluoresces when calcium is present, as seen in the graphic below.

![Graphic visualizing calcium imaging in a population of neurons](http://neurofinder.codeneuro.org/components/assets/movie.gif)

In a neuron, calcium is closely related to spiking behavior, but inferring the spikes from calcium fluorescences is a non-trivial task. There are often sources of noise in the recordings, and the relationship between the amount of calcium in the cell and the cell's membrane voltage is non-linear and time-dependent.

In this tutorial, we will use recent methods from deep learning to characterize the distribution of calcium fluorescences that correspond to a spike. In other words, we will capture the "noise" of a recording. In this process, we will build up our understand of deep learning methods which can be applied to other tasks.

## Preparation

This tutorial assumes you have an idea of the fundamentals of deep learning. We will use the deep learning framework [Keras](https://keras.io/) to build our models. Take a look through the documentation to understand how things work.

The tutorial will use the training data provided from the [SpikeFinder competition](http://spikefinder.codeneuro.org/). This data is hosted on AWS [here](https://s3.amazonaws.com/neuro.datasets/challenges/spikefinder/spikefinder.train.zip) and can be loaded and managed using Pandas.

More resources for Generative Adversarial Networks can be found [here](http://gandlf.org/background/).
