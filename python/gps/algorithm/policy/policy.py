""" This file defines the base class for the policy. """
import abc


class Policy(object):
    """ Computes actions from states/observations. """
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def act(self, x, obs, t, noise, usescale):
        """
        Args:
            x: State vector.
            obs: Observation vector.
            t: Time step.
            noise: A dU-dimensional noise vector.
        Returns:
            A dU dimensional action vector.
        """
        raise NotImplementedError("Must be implemented in subclass.")
    
    def reset(self):
        return
    
    # Called when done using the object - must call reset() before starting to use it again
    def finalize(self):
        return
    
    def set_meta_data(self, meta):
        """
        Set meta data for policy (e.g., domain image, multi modal observation sizes)
        Args:
            meta: meta data.
        """
        return
