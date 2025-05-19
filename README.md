# Timeseries Saliency Maps: Explaining models across multiple domains

Official Pytorch/Tensorflow implementation of Cross-Domain Saliency Maps.
The method does not require any model model retraining or modications.

<img src="./figures/cross_domain_saliency_maps_banner.svg" width="755">

# Installation
Download this repository:
```
git clone https://github.com/esl-epfl/cross-domain-saliency-maps
```

Install using ```pip```:
```
pip install ./cross_domain_saliency_maps
```

# Usage
Detailed examples for using with pytorch and  models can be 
found in the corresponding [examples](./examples/):
1. [Pytorch implementation](./examples/torch_demo.ipynb)
2. [Tensorflow implementation](./examples/tensorflow_demo.ipynb)

The library supports generating saliency maps for any domain which
can be formulated as an invertible transformation with a differentiable
inverse transformation. 

To generate maps expressed in a domain, a corresponding ```Domain```
object needs to be defined. This describes the operations performed
during the forward and inverse transformations. 

Implementations for the [Frequency and Independent Component Analysis (ICA)](#saliency-maps-in-the-frequency-and-ica-domains)
transformations are already implemented and can be directly deployed. 
Additionally, the libraryprovides the flexibility of 
[defining new transformations](#saliency-maps-in-any-domain).

## Saliency Maps in the Frequency and ICA domains
The following domains are already implemented and can be
directly used to generate saliency maps:

1. **Frequency Domain.** Each point in the map corresponds to
the importance of the corresponding frquency component. The 
Fourier transform is used to transform the time-domain to 
the frequency domain. The corresponding ```Domain``` object
is ```FourierDomain```. The map can be directly generated:
```fourierIG = FourierIG(model, n_iterations, output_channel = 0)``` 

2. **Independent Component Domain.** Each point in the 
map corresponds to an independent component (IC) of the ICA
decomposition. Any ICA implementation can be used as long as it
complies with [```sklearn.decomposition.FastICA```](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html). The domain is defined 
by ```ICADomain```. Before generating the map a ```FastICA```
needs to be fitted to the input sample (see [example](./examples/tensorflow_demo.ipynb)). The map can be directly generated:
``` icaIG = ICAIG(model, fastICA, n_iterations, output_channel = 0)``` 

## Saliency Maps in any domain
The library supports extending the Cross-domain Integrated Gradients
for any invertible domain with a differentiable inverse transform. This
requires:
1. Creating the propert ```Domain``` object describing the corresponding
transform. ```Domain``` objects need to inherit from ```DomainBase``` and
implement the required functions. More details can be found in the 
implementation of the ```FourierDomain``` and ```ICADomain``` (
[tensorflow](/src/cross_domain_saliency_maps/tensorflow_ig/domain_transforms.py), [pytorch](/src/cross_domain_saliency_maps/torch_ig/domain_transforms.py)).

2. Calling ```CrossDomainIG``` with the new domain as the input. This
can be done either by creating a ```CrossDomainIG```, initializing it
with the new domain, or by implementing a new dedicated class inheriting
```CrossDomainIG```. For more details check the implementations of 
```FourierIG``` and ```ICAIG```(
[tensorflow](/src/cross_domain_saliency_maps/tensorflow_ig/cross_domain_integrated_gradients.py), [pytorch](/src/cross_domain_saliency_maps/torch_ig/cross_domain_integrated_gradients.py)).

# Reference
[TODO reference]