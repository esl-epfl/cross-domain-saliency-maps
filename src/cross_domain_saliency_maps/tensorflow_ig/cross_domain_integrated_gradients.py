import tensorflow as tf
import numpy as np

from cross_domain_saliency_maps.tensorflow_ig.domain_transforms import DomainBase
from cross_domain_saliency_maps.tensorflow_ig.domain_transforms import FourierDomain
from cross_domain_saliency_maps.tensorflow_ig.domain_transforms import ICADomain

class CrossDomainIG:
    """ Cross Domain IG base class. Defines the basic functionality 
        for cross-domain ig. 

        Attributes:
        model (tf.keras.models.Model): A tensorflow model.
        n_iterations (int): Number of iterations for approximating IG.
        output_channel(int): The output channel of the model for which 
                             we generate the saliency map.
        dtype (dtype): The type of the target domain.
    """
    def __init__(self, model: tf.keras.models.Model, 
                 n_iterations: int, 
                 output_channel: int, 
                 dtype = tf.float32):
        """ Initializes CrossDomainIG.

        Args:
            model (tf.keras.models.Model): A tensorflow model for which 
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
        
        with tf.GradientTape() as tape:
            X_in = self.domain.get_coefficients()
            X_baseline = self.domain.get_coefficient_baseline()

            a = tf.constant(np.linspace(0, 1, self.n_iterations), dtype = X_in.dtype)

            X_samples = X_baseline + (X_in - X_baseline) * a[:, tf.newaxis, tf.newaxis]
            tape.watch(X_samples)
            x_ = self.domain.inverse_transform(X_samples)
            y_ = self.model(x_)
            grads = tape.gradient(y_[:, self.output_channel], X_samples)
            
        S = tf.math.reduce_mean(tf.math.conj(grads), axis = 0)
        self.multiIG = tf.math.real((X_in[0, :] - X_baseline[0, :]) * S)
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
                 model: tf.keras.models.Model, 
                 n_iterations: int, 
                 output_channel: int, 
                 dtype = tf.float32):
        super().__init__(model, n_iterations,
                         output_channel, dtype)
        
        self.initialize_domain(FourierDomain, dtype = dtype)

class ICAIG(CrossDomainIG):
    """ Implementation of the CrossDomainIG specifically for the
    target domain defined by the Independent Component Analysis
    (ICA) decomposition.

    Here the decomposed signal channels act as the signal basis,
    while the mixing matrix is the features on which the saliency
    is expressed.

    Args:
        model (tf.keras.models.Model): The tensorflow model.
        ica: A ICA object compatible with sklearn's FastICA object.
        n_iterations (int): Number of steps to approximate the integral
            in the IG.
        output_channel (int): The model's output channel for which the
            saliency map will be generated.
        dtype: The type of the features. 
    """
    def __init__(self, 
                 model: tf.keras.models.Model, 
                 ica, 
                 n_iterations: int, 
                 output_channel: int, 
                 dtype = tf.float32):
        super().__init__(model, n_iterations,
                         output_channel, dtype)
        
        self.initialize_domain(ICADomain, ica = ica, dtype = dtype)
    
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
        self.multiIG = tf.reduce_sum(self.multiIG, axis = 1)
        return self.multiIG