import jax.numpy as jnp
from ..utils import cross_entropy_loss
from jax import jit
from abc import ABC, abstractmethod
from .conv import JaxOptimized as Conv
from .rnncell import JaxOptimized as Rnn
from .fc import dropout


@jit
def _acc(y_true_proba, y_pred_proba):
    '''
    Computes the accuracy of predictions.

    Args:
        y_true_proba: True labels (one-hot encoded) of shape (batch_size, num_classes).
        y_pred_proba: Predicted probabilities of shape (batch_size, num_classes).

    Returns:
        Accuracy as a scalar value.
    '''

    y_true = jnp.argmax(y_true_proba, axis=1)
    y_pred = jnp.argmax(y_pred_proba, axis=1)
    return jnp.mean(y_true == y_pred)


class NNModel(ABC):
    '''
    An abstract base class for defining, training, and evaluating neural network models.

    This class provides a framework for building neural network models, including support for
    optimization, dropout, and logging during training. Subclasses must implement the
    `predict_proba` method to define the model's forward pass.

    Attributes:
        epoches (int): Number of training epochs (default: 100).
        log_wise (int): Frequency of logging during training (default: 10).
        optr (Optimizer): Optimizer instance (e.g., Adam, SGD). Must be set by subclasses.
        train (bool): Boolean flag indicating whether the model is in training mode.

    Methods:
        __init__: Initializes the NNModel class.
        predict_proba: Abstract method for making predictions using the model.
        fit: Trains the model on the provided training data and evaluates it on the test data.

    Examples
    --------
    ```python
        # Define a subclass of NNModel
        class MyModel(NNModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Initialize model parameters and optimizer
                self.optr = Adam(params, lr=self.lr, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon)

            def predict_proba(self, x, params, train=True):
                # Define the forward pass of the model
                pass

        # Create an instance of the model
        model = MyModel(lr=0.001, epoches=50)

        # Train the model
        acc, loss, tacc, tloss = model.fit(x_train, y_train_proba, x_test, y_test_proba)
    ```
    '''

    def __init__(self,
                 lr=0.01,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-6,
                 epoches=100,
                 log_wise=10):
        '''
        Initializes the NNModel class.

        Args:
            lr: Learning rate for the optimizer (default: 0.01).
            beta1: Beta1 parameter for the optimizer (default: 0.9).
            beta2: Beta2 parameter for the optimizer (default: 0.999).
            epsilon: Small constant for numerical stability (default: 1e-6).
            epoches: Number of training epochs (default: 100).
            log_wise: Frequency of logging (default: 10)
        '''

        self.epoches = epoches
        self.log_wise = log_wise
        self.optr = None

    @abstractmethod
    def predict_proba(self, x: jnp.ndarray, params, train=True):
        '''
        Abstract method for making predictions using the model.

        Args:
            x: Input data of shape (batch_size, ...).
            params: Model parameters.
            train: Whether the model is in training mode (default: True).

        Returns:
            Predicted probabilities for each class.
        '''

        pass

    def fit(self, x_train, y_train_proba, x_test, y_test_proba, epoches=100):
        '''
        Trains the model on the provided training data and evaluates it on the test data.

        Args:
            x_train: Training input data of shape (batch_size, ...).
            y_train_proba: Training labels (one-hot encoded) of shape (batch_size, num_classes).
            x_test: Test input data of shape (batch_size, ...).
            y_test_proba: Test labels (one-hot encoded) of shape (batch_size, num_classes).
            epoches: Number of training epochs (default: 100).

        Returns:
            acc: List of training accuracy values for each epoch.
            loss: List of training loss values for each epoch.
            tacc: List of test accuracy values for each epoch.
            tloss: List of test loss values for each epoch.
        '''

        cnt = 0

        _loss = lambda params, x, y_true: cross_entropy_loss(y_true, self.predict_proba(x, params, True))
        _loss = jit(_loss)  # accelerate loss function by JIT
        self.optr.open(_loss, x_train, y_train_proba)

        _tloss = lambda params: cross_entropy_loss(y_test_proba, self.predict_proba(x_test, params, False))
        _tloss = jit(_tloss)  # accelerate loss function by JIT

        acc, loss, tacc, tloss = [], [], [], []  # train acc, train loss, test acc, test loss

        for _ in range(epoches):
            loss.append(_loss(self.optr.get_params(), x_train, y_train_proba))
            tloss.append(_tloss(self.optr.get_params()))

            self.train = True  # use dropout only while updating grads
            self.optr.update()
            self.train = False

            acc.append(_acc(y_train_proba, self.predict_proba(x_train, self.optr.get_params())))
            tacc.append(_acc(y_test_proba, self.predict_proba(x_test, self.optr.get_params())))
            cnt += 1
            if cnt % 10 == 0:
                print(f'>> epoch: {cnt}, train acc: {acc[-1]}, train loss: {loss[-1]}; test acc: {tacc[-1]}, test loss: {tloss[-1]}')

        return acc, loss, tacc, tloss


__all__ = ['Conv', 'Rnn', 'NNModel', 'dropout']
