# calcium-gan

Tutorial on building a GAN model to calcium imaging.

## Abstract

A key idea in neuroscience is that neurons encode senses and behavior with "spikes" of electrical activity. These spikes initiate in the cell body and propagate down the wire-like axon to synapses, i.e., connections with other neurons. These neurons integrate the signal with many other incoming spikes from many other neurons. For nearly a century, the most common way of recording spikes has been with extracellular elecrodes. This type of electrode is basically a wire that in close proximity to a neuron can record the electrical potential across the membrane--like a tiny microphone.

While single-unit recordings of extracellular activity have given us important insights into the neural code, they do not paint a complete picture. Importantly they do not allow us to easily see what might be encoded at a population level. Multielectrode arrays can help with this but even with arrays we may miss important information about the population, e.g., which types of neurons in a region connect with each other and which do not spike at all during a given behavior. 

One alternative to recording neurons with electrodes is imaging calcium concentration in the cell. It is known that calcium levels increase when a neuron spikes. Calcium indicator dyes bind to calcium and upon doing so, the indicator molecule adopts a conformation that is fluorescent. A gif of what this looks like is shown below.

![Graphic visualizing calcium imaging in a population of neurons](http://neurofinder.codeneuro.org/components/assets/movie.gif)

Inferring spikes from fluorescence is a non-trivial task. The relationship between the amount of calcium in the cell and the cell's membrane potential is non-linear and time-dependent--calcium levels do not necessarily track membrane voltage closely . Binding of the indicator dyes to calcium is itself a reaction that takes time to occur. In addition there are often sources of noise in the recordings and the results can depend on the experimenter's choice of the region of a neuron to image (although most protocols attempt to minimize noise due to "region of interest").

In this tutorial, we will use recent methods from deep learning to characterize the distribution of  fluorescence signals that correspond to spikes and to non-spiking periods. In this process, we will build up our understand of deep learning methods which can be applied to other tasks.

## Preparation

This tutorial assumes you have an idea of the fundamentals of deep learning. We will use the deep learning framework [Keras](https://keras.io/) to build our models. Take a look through the documentation to understand how things work.

The tutorial will use the training data provided from the [SpikeFinder competition](http://spikefinder.codeneuro.org/). This data is hosted on AWS [here](https://s3.amazonaws.com/neuro.datasets/challenges/spikefinder/spikefinder.train.zip) and can be loaded and managed using Pandas.

More resources for Generative Adversarial Networks can be found [here](http://gandlf.org/background/).
