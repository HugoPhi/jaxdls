from abc import ABC, abstractmethod
from .utils import cross_entropy_loss


class Loss(ABC):

    def __init__(self, f):
        '''
        f: x, params, train -> y_proba
        '''
        self.f = f

    @abstractmethod
    def get_loss(self, train):
        '''
        loss function: params, x, y_true -> loss
        '''
        pass

    @abstractmethod
    def get_embed_loss(self, x, y_true, train):
        '''
        embed loss funtion: params -> loss
        '''
        pass


class CrossEntropyLoss(Loss):

    def __init__(self, f):
        super().__init__(f)

    def get_loss(self, train):
        loss_function = lambda params, x, y_true: cross_entropy_loss(y_true, self.f(x, params, train))
        return loss_function

    def get_embed_loss(self, x, y_true, train):
        embed_loss_function = lambda params: cross_entropy_loss(y_true, self.f(x, params, train))
        return embed_loss_function
