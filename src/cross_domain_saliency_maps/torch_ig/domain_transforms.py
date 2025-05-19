import abc
import torch
import numpy as np

class DomainBase:
    """ Defines the base class for a domain transformation.

    Attributes:
        device (torch.device): The device used to run the model.
        coefficients (tf.Tensor): The input samples expressed
            in the target domain. 
        baseline_coefficients (tf.Tensor): The baseline samples expressed
            in the target domain. 
        dtype: The dtype of the coefficients when they are 
            converted to tf.Tensor.
    """
    def __init__(self, device: torch.device, dtype = torch.float32):
        """ Initialize the Domain.
        """
        self.device = device
        self.dtype = dtype
        self.coefficients = None
        self.baseline_coefficients = None
            
    def get_coefficients(self):
        """ Return the coeffcients.
        """
        return self.coefficients
    
    def get_coefficient_baseline(self):
        """ Return the baseline coefficients.
        """
        return self.baseline_coefficients

    @abc.abstractmethod
    def set_coefficients(self, x, x_baseline):
        """ Sets the coefficients depending on the transform
        used. Should be implemented by class inheriting DomainBase.

        Args:
            x (np.array): The input sample.
            x_baseline (np.array): The baseline sample.
        """
        return
    
    @abc.abstractmethod
    def forward_transform(self, x):
        """ Performs the forward transform, transforming the input
        sample x into the corresponding target domain.

        Args:
            x : The input sample
        """
        return
    
    @abc.abstractmethod
    def inverse_transform(self, x_input):
        """ Performs the inverse transform, transforming the
        sample x from the target domain back to the original one.

        Args:
            x_input : The input sample
        """
        return

class FourierDomain(DomainBase):
    """ Domain implementation for the Fourier transform, mapping 
    time-domain samples into the frequency domain.
    """
    def __init__(self, device, dtype = torch.float32, time_dimension = -1):
        """ Initialize the Fourier Domain.

        Args:
            dtype: The type of the input features. 
            time_dimension: The dimension in the input which 
                corresponds to the time-points.
        """
        super().__init__(device = device, dtype = dtype)
        self.time_dimension = time_dimension
    
    def forward_transform(self, x: torch.Tensor):
        """ Implementation of the forward transform, transforming the input
        time sample to the corresponding frequency domain sample. 

        Args:
            x (tf.Tensor): Input time-domain sample.
        """
        return torch.fft.fft(x.type(torch.complex64), dim = self.time_dimension)
      
    def set_coefficients(self, x: np.array, x_baseline: np.array):
        """ Sets the frequency coefficients transforming the input and
        baseline samples into the frequency domain.

        Args:
            x (np.array): The input sample in time-domain.
            x_baseline (np.array): The baseline sample in time-domain.
        """
        x_tf = torch.from_numpy(x).type(self.dtype).to(self.device)
        x_baseline_tf = torch.from_numpy(x_baseline).type(self.dtype).to(self.device)

        self.coefficients = self.forward_transform(x_tf)
        self.baseline_coefficients = self.forward_transform(x_baseline_tf)
    
    def inverse_transform(self, x_input: torch.Tensor):
        """ Inverse transform, transforming the frequency domain input
        x_input points back into the time domain.

        Args:
            x_input (tf.Tensor): The frequency domain input.
        """
        return torch.fft.ifft(x_input, dim = self.time_dimension).to(torch.float32)

class ICADomain(DomainBase):
    """ Implements the Domain for the Independent Component Analysis 
    (ICA) decomposition.

    We consider the independent channels to form the basis of 
    the input. The mixing matrix forms the actual input coefficients. 
    This way, ICA IG expresses significance of each independent component.

    """
    def __init__(self, device, ica, dtype = torch.float32, channel_permutation = (0, 2, 1)):
        super().__init__(device = device, dtype = dtype)
        self.channel_permutation = channel_permutation
        self.ica = ica
    
    def forward_transform(self, x: np.array):
        """ Forward transform, transforms the input channels into 
        independent channels.

        Args:
            x (np.array): Input sample
        """
        X = self.ica.transform(x)
        return X
    
    def set_coefficients(self, x:np.array, x_baseline: np.array):
        """ Sets the coefficients in the ICA space. We consider the 
        independent channels to form the basis of the input. The 
        mixing matrix forms the actual input coefficients. This way
        ICA IG expresses significance of each independent component.

        Since the basis of the transform are the independent components,
        we only consider zero baseline.

        Args:
            x (np.array): Input sample of size [1, n_time_points, n_channels]
        """

        self.basis = torch.from_numpy(self.forward_transform(x[0, ...].T)).type(self.dtype).to(self.device)

        self.coefficients = torch.from_numpy(self.ica.mixing_.T).type(self.dtype).to(self.device)[None, ...]
        self.baseline_coefficients = torch.zeros_like(self.coefficients)
        self.mean = torch.from_numpy(self.ica.mean_).type(self.dtype).to(self.device)
    
    def inverse_transform(self, x_input: torch.Tensor):
        """ Inverses the ICA transform given an input matrix 
        x_input and the basis stored in the domain. 

        Args:
            x_input (torch.Tensor): Input unmixing matrix of size [n_channels, n_channels].
        """
        return torch.transpose(torch.matmul(self.basis, x_input) + self.mean, 1, 2)