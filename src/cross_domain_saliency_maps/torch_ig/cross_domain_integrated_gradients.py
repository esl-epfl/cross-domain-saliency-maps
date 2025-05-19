import torch
import numpy as np

from cross_domain_saliency_maps.torch_ig.domain_transforms import DomainBase
from cross_domain_saliency_maps.torch_ig.domain_transforms import FourierDomain
from cross_domain_saliency_maps.torch_ig.domain_transforms import ICADomain

from tqdm import tqdm

class CrossDomainIG:
    """ Cross Domain IG base class. Defines the basic functionality 
        for cross-domain ig. 

        Attributes:
        model (torch.nn.Module): A pytorch model.
        n_iterations (int): Number of iterations for approximating IG.
        output_channel(int): The output channel of the model for which 
                             we generate the saliency map.
        dtype (dtype): The type of the target domain.
    """
    def __init__(self, model: torch.nn.Module, 
                 n_iterations: int, 
                 output_channel: int, 
                 dtype = torch.float32):
        """ Initializes CrossDomainIG.

        Args:
            model (torch.nn.Module): A pytorch model for which 
                saliency maps will be generated.
            n_iterations (int): Number of steps used in approximating
                the integral in the Integrated Gradients computation.
            output_channel (int): The channle of the model's output used
                for the saliency map (e.g. the class channel).
        """
        self.model = model
        self.n_iterations = n_iterations
        self.output_channel = output_channel

    def initialize_domain(self, Domain: DomainBase, **kwargs):
        """ Initializes the target domain in which the IG is 
            expressed.

            Args:
                Domain (DomainBase): The target domain.
                **kwargs: Parameters needed to initialize the Domain.
        """
        self.domain = Domain(**kwargs)
    
    def run(self, x: np.array, x_baseline: np.array):
        """ Runs the saliency map generation for input sample x
            using the x_baseline for baseline.

            Args:
                x (np.array): a single sample with shape [1, n_timesteps, n_channels]
                x_baseline (np.array): the baseline sample with shape [1, n_timesteps, n_channels]
        """
        self.domain.set_coefficients(x, x_baseline)

        grad_sum = 0

        X_in = self.domain.get_coefficients()
        X_baseline = self.domain.get_coefficient_baseline()

        X_samples = [ X_baseline + (float(i) / self.n_iterations) * (X_in - X_baseline) for i in range(1, self.n_iterations + 1)]

        for X_sample in tqdm(X_samples):
            X_sample.requires_grad = True
            x_ = self.domain.inverse_transform(X_sample)
            prediction = self.model(x_)
            prediction[0, self.output_channel].backward()
            grad_sum += torch.conj(X_sample.grad)

        grad_sum /= self.n_iterations
        self.multiIG = torch.real((X_in - X_baseline) * grad_sum)
        
        return self.multiIG

    def getMultiIG(self):
        """ Get the generated saliency map.
        """
        return self.multiIG


class FourierIG(CrossDomainIG):
    """ Implementation of the CrossDomainIG specifically for the
    frequency target domain. 
    """
    def __init__(self, 
                 model: torch.nn.Module, 
                 n_iterations: int, 
                 output_channel: int, 
                 device: torch.device,
                 dtype = torch.float32):
        super().__init__(model, n_iterations,
                         output_channel, dtype)
        
        self.initialize_domain(FourierDomain, device = device, dtype = dtype)

class ICAIG(CrossDomainIG):
    """ Implementation of the CrossDomainIG specifically for the
    target domain defined by the Independent Component Analysis
    (ICA) decomposition.

    Here the decomposed signal channels act as the signal basis,
    while the mixing matrix is the features on which the saliency
    is expressed.

    Args:
        model (torch.nn.Module): The pytorch model.
        ica: A ICA object compatible with sklearn's FastICA object.
        n_iterations (int): Number of steps to approximate the integral
            in the IG.
        output_channel (int): The model's output channel for which the
            saliency map will be generated.
        dtype: The type of the features. 
    """
    def __init__(self, 
                 model: torch.nn.Module, 
                 ica, 
                 n_iterations: int, 
                 output_channel: int, 
                 device: torch.device,
                 dtype = torch.float32):
        super().__init__(model, n_iterations,
                         output_channel, dtype)
        
        self.initialize_domain(ICADomain, ica = ica, dtype = dtype, device = device)
    
    def run(self, x: np.array, x_baseline: np.array):
        """ Runs the IG generation. 

        The IG components are expressed over the elements of 
        the entire unmixing matrix, since this is considered
        as the input features (projected over the basis defined
        by the independent channels). We sume along the first
        dimension of the matrix to express the IG in terms of 
        independent channels.

        Args:
            x (np.array): Input sample for which to generate saliency.
            x_basleine (np.array): Baseline sample.
        """
        super().run(x, x_baseline)
        self.multiIG = torch.sum(self.multiIG[0], dim = 1)
        return self.multiIG