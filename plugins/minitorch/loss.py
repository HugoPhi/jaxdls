from abc import ABC, abstractmethod
from jax import jit

from .utils import cross_entropy_loss


class Loss(ABC):
    def __init__(self):
        pass
