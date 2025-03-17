'''
JAX Optimization Algorithms Module

* Last Updated: 2025-03-15
* Author: HugoPhi, [GitHub](https://github.com/HugoPhi)
* Maintainer: hugonelsonm3@gmail.com

This module provides a collection of optimization algorithms implemented in JAX,
including Gradient Descent, Momentum, Nesterov Accelerated Gradient, AdaGrad,
RMSProp, AdaDelta, and Adam. These optimizers are designed to work seamlessly
with JAX's functional programming paradigm and automatic differentiation.

Key Features:
    - Abstract base class `Optimizter` for defining custom optimizers
    - Implementations of popular optimization algorithms:
        - Gradient Descent (RawGD)
        - Momentum
        - Nesterov Accelerated Gradient (Nesterov)
        - AdaGrad
        - RMSProp
        - AdaDelta
        - Adam
    - Support for batch processing and training state management
    - Integration with JAX's JIT compilation for performance optimization

Structure:
    - Abstract `Optimizter` class with core functionality for managing parameters and training state
    - Concrete optimizer classes implementing specific algorithms
    - Batch processing utilities for efficient training

Typical Usage:
    1. Initialize an optimizer with model parameters and hyperparameters
    2. Open the optimizer with training data and a loss function
    3. Update parameters using the optimizer's `update` method
    4. Close the optimizer when training is complete

Note: All optimizers assume the use of JAX's functional programming paradigm and
require proper management of model parameters and optimizer states.
'''

import jax.numpy as jnp
from jax import grad, tree, lax, random
from abc import ABC, abstractmethod


class Optimizter(ABC):
    '''
    An abstract base class for optimization algorithms. It serves as a container for model parameters
    and provides methods to update the parameters during training. This class is designed to be
    inherited by specific optimization algorithms (e.g., SGD, Adam, etc.).

    The optimizer manages the training process by:
    1. Storing model parameters.
    2. Updating parameters using gradients computed from a loss function.
    3. Handling batch processing and training state (e.g., open/close).

    Subclasses must implement the abstract methods `__init__`, `update`, and `flash`.

    Attributes:
        params: The model parameters to be optimized.
        lr: Learning rate (if applicable to the optimizer).
        batch_size: Size of each training batch.
        is_open: A boolean indicating whether the optimizer is in an active state (open for training).
        steps: The number of optimization steps taken so far.
    '''

    @abstractmethod
    def __init__(self):
        '''
        Initializes the optimizer. Subclasses must implement this method to set up optimizer-specific
        attributes (e.g., learning rate, momentum, etc.) and initialize the model parameters.
        '''
        self.is_open = False
        pass

    @abstractmethod
    def update(self):
        '''
        Updates the model parameters using the gradients computed from the loss function.
        Subclasses must implement this method to define the specific optimization algorithm.
        '''
        pass

    @abstractmethod
    def flash(self):
        '''
        Resets the optimizer's internal state (e.g., momentum buffers, gradient accumulators).
        This method is called when the optimizer is opened or re-initialized.
        Subclasses must implement this method.
        '''
        pass

    def open(self, loss_function, x_train: jnp.ndarray, y_train: jnp.ndarray, short_batch='drop', key=None):
        '''
        Prepares the optimizer for training by initializing its state and setting up the training data.

        Args:
            loss_function: A loss function that computes the scalar loss given model parameters,
                          input data, and true labels. It must be JIT-compiled.
                          Signature: `f(params, x, y_true) -> scalar`.
            x_train: Input data for training. Shape: `(num_samples, ...)`.
            y_train: True labels for training. Shape: `(num_samples, ...)`.
            short_batch: The Strategy to handle short batch. including:
                - 'drop': drop short batch, used when: dataset size >> batch size, num_batches = N // B
                - 'pad': append arr[-B:] to trimmed arr, used when: dataset size >~ batch size, num_batches = N // B + 1
            key: A random number generator key used for initialization.

        Notes:
            - The training data is divided into batches based on the `batch_size` attribute.
            - If the dataset size is not divisible by `batch_size`, the last batch is dropped.
            - The optimizer must be opened before calling `update` or accessing parameters.
        '''

        if self.is_open is True:
            print('oprimizer is already opened.')
        else:
            self.is_open = True

            self.flash()
            self._loss = loss_function
            self.key = key

            if short_batch == 'drop':
                self.num_batches = x_train.shape[0] // self.batch_size
                trimmed_size = self.num_batches * self.batch_size
                self.x_train = x_train[:trimmed_size]
                self.y_train = y_train[:trimmed_size]
            elif short_batch == 'pad':
                self.num_batches = x_train.shape[0] // self.batch_size
                trimmed_size = self.num_batches * self.batch_size
                self.x_train = jnp.concatenate((x_train[:trimmed_size], x_train[-self.batch_size:]), axis=0)  # pad last batch
                self.y_train = jnp.concatenate((y_train[:trimmed_size], y_train[-self.batch_size:]), axis=0)
                self.num_batches += 1
            else:
                raise ValueError(f'short_batch must be in "drop" or "pad", but get {short_batch} instead.')

            print(f'[*] oprimizer opened with {self.num_batches} batches with batch size {self.batch_size}.')

    def close(self):
        '''
        Closes the optimizer, preventing further updates or parameter access until it is reopened.
        '''

        if self.is_open is False:
            print('oprimizer is already closed.')
        else:
            self.is_open = False

    def get_params(self):
        '''
        Returns the current model parameters.

        Returns:
            The model parameters being optimized.
        '''

        return self.params

    def get_steps(self):
        '''
        Returns the number of optimization steps taken so far.

        Returns:
            The number of steps (integer).
        '''

        return self.steps


class Adam(Optimizter):
    '''
    Adam (Adaptive Moment Estimation) optimizer, a popular stochastic optimization algorithm
    that combines the benefits of Momentum and RMSProp. It adapts the learning rate for each
    parameter using estimates of the first and second moments of the gradients.

    Attributes:
        params: The model parameters to be optimized.
        lr: Learning rate (default: 0.01).
        beta1: Exponential decay rate for the first moment estimates (default: 0.9).
        beta2: Exponential decay rate for the second moment estimates (default: 0.999).
        epsilon: Small constant for numerical stability (default: 1e-6).
        batch_size: Size of each training batch (default: 32).
        V: First moment vector (momentum-like term).
        VV: Second moment vector (RMSProp-like term).
        steps: Number of optimization steps taken so far.
        is_open: Boolean indicating whether the optimizer is active.
    '''

    def __init__(self, params,
                 lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-6,
                 batch_size=32):
        '''
        Initializes the Adam optimizer.

        Args:
            params: Model parameters to be optimized.
            lr: Learning rate (default: 0.01).
            beta1: Exponential decay rate for the first moment estimates (default: 0.9).
            beta2: Exponential decay rate for the second moment estimates (default: 0.999).
            epsilon: Small constant for numerical stability (default: 1e-6).
            batch_size: Size of each training batch (default: 32).
        '''

        super().__init__()

        self.params = params

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.batch_size = batch_size

    def flash(self):
        '''
        Resets the optimizer's internal state, including moment vectors and step count.
        This method is called when the optimizer is opened or re-initialized.
        '''

        self.V = tree.map(lambda x: jnp.zeros_like(x), self.params)
        self.VV = tree.map(lambda x: jnp.zeros_like(x), self.params)

        self.steps = 0

    def update(self):
        '''
        Performs a single optimization step using the Adam algorithm. It updates the model
        parameters based on the gradients computed from the loss function.

        Raises:
            ValueError: If the optimizer is not open.
        '''

        if self.is_open is False:
            raise ValueError('please open optimizer first!!!')
        else:
            ixs = jnp.arange(self.num_batches)
            bxs = self.x_train.reshape(self.num_batches, self.batch_size, *self.x_train.shape[1:])
            bys = self.y_train.reshape(self.num_batches, self.batch_size, *self.y_train.shape[1:])

            if self.key is not None:
                subkeys = random.split(self.key, self.num_batches + 1)
                self.key, subkeys = subkeys[0], subkeys[1:]  # update self.key & get subkeys

            def one_batch(carry, ix):

                def adam(d_w, w, v, vv):
                    # t = jnp.clip(carry['steps'], 0, 1e3) + 1
                    t = carry['steps'] + 1

                    new_v = self.beta1 * v + (1 - self.beta1) * d_w
                    new_vv = self.beta2 * vv + (1 - self.beta2) * d_w * d_w

                    v_hat = new_v / (1 - self.beta1**t)
                    vv_hat = new_vv / (1 - self.beta2**t)
                    step = - self.lr * v_hat / (jnp.sqrt(vv_hat) + self.epsilon)

                    new_w = w + step
                    return jnp.stack((
                        new_w,
                        new_v,
                        new_vv,
                    ))

                bx = bxs[ix]
                by = bys[ix]

                if self.key is not None:
                    kkey = subkeys[ix]
                    d_params = grad(self._loss, argnums=0)(carry['params'], bx, by, kkey)
                else:
                    d_params = grad(self._loss, argnums=0)(carry['params'], bx, by)

                pack = tree.map(adam, d_params, carry['params'], carry['V'], carry['VV'])  # use Adam
                carry['params'] = tree.map(lambda x: x[0], pack)
                carry['V'] = tree.map(lambda x: x[1], pack)
                carry['VV'] = tree.map(lambda x: x[2], pack)
                carry['steps'] += 1

                return carry, None

            pack, _ = lax.scan(one_batch, {
                'params': self.params,
                'V': self.V,
                'VV': self.VV,
                'steps': self.steps,
            }, ixs)

            self.params, self.V, self.VV, self.steps = pack['params'], pack['V'], pack['VV'], pack['steps']


class RawGD(Optimizter):
    '''
    A simple Gradient Descent (GD) optimizer. This optimizer updates model parameters
    by moving them in the direction of the negative gradient, scaled by a learning rate.

    Attributes:
        params: The model parameters to be optimized.
        lr: Learning rate (default: 0.01).
        batch_size: Size of each training batch (default: 32).
        steps: Number of optimization steps taken so far.
        open: Boolean indicating whether the optimizer is active.
    '''

    def __init__(self, params,
                 lr=0.01, batch_size=32):
        '''
        Initializes the Gradient Descent optimizer.

        Args:
            params: Model parameters to be optimized.
            lr: Learning rate (default: 0.01).
            batch_size: Size of each training batch (default: 32).
        '''

        super().__init__()

        self.params = params

        self.lr = lr

        self.batch_size = batch_size

    def flash(self):
        '''
        Resets the optimizer's internal state, including the step count.
        This method is called when the optimizer is opened or re-initialized.
        '''

        self.steps = 0
        self.is_open = False

    def update(self):
        '''
        Performs a single optimization step using Gradient Descent. It updates the model
        parameters based on the gradients computed from the loss function.

        Raises:
            ValueError: If the optimizer is not open.
        '''

        if self.is_open is False:
            raise ValueError('please open optimizer first!!!')
        else:
            ixs = jnp.arange(self.num_batches)
            bxs = self.x_train.reshape(self.num_batches, self.batch_size, *self.x_train.shape[1:])
            bys = self.y_train.reshape(self.num_batches, self.batch_size, *self.y_train.shape[1:])

            if self.key is not None:
                subkeys = random.split(self.key, self.num_batches + 1)
                self.key, subkeys = subkeys[0], subkeys[1:]  # update self.key & get subkeys

            def one_batch(carry, ix):

                def gd(d_w, w):
                    new_w = w - self.lr * d_w
                    return new_w

                bx = bxs[ix]
                by = bys[ix]

                if self.key is not None:
                    kkey = subkeys[ix]
                    d_params = grad(self._loss, argnums=0)(carry['params'], bx, by, kkey)
                else:
                    d_params = grad(self._loss, argnums=0)(carry['params'], bx, by)

                pack = tree.map(gd, d_params, carry['params'])
                carry['params'] = pack
                carry['steps'] += 1

                return carry, None

            pack, _ = lax.scan(one_batch, {
                'params': self.params,
                'steps': self.steps,
            }, ixs)

            self.params, self.steps = pack['params'], pack['steps']


class Momenum(Optimizter):
    '''
    Momentum optimizer, a variant of Gradient Descent that incorporates momentum to accelerate
    convergence and reduce oscillations. It accumulates a velocity vector in the direction of
    the gradient and uses it to update the parameters.

    Attributes:
        params: The model parameters to be optimized.
        lr: Learning rate (default: 0.01).
        beta: Momentum factor (default: 0.9).
        batch_size: Size of each training batch (default: 32).
        V: Velocity vector (accumulated gradients).
        steps: Number of optimization steps taken so far.
        open: Boolean indicating whether the optimizer is active.
    '''

    def __init__(self, params,
                 lr=0.01, beta=0.9,
                 batch_size=32):
        '''
        Initializes the Momentum optimizer.

        Args:
            params: Model parameters to be optimized.
            lr: Learning rate (default: 0.01).
            beta: Momentum factor (default: 0.9).
            batch_size: Size of each training batch (default: 32).
        '''

        super().__init__()

        self.params = params

        self.lr = lr
        self.beta = beta

        self.batch_size = batch_size

    def flash(self):
        '''
        Resets the optimizer's internal state, including the velocity vector and step count.
        This method is called when the optimizer is opened or re-initialized.
        '''

        self.V = tree.map(lambda x: jnp.zeros_like(x), self.params)

        self.steps = 0
        self.is_open = 0

    def update(self):
        '''
        Performs a single optimization step using Momentum. It updates the model parameters
        based on the gradients computed from the loss function and the accumulated velocity.

        Raises:
            ValueError: If the optimizer is not open.
        '''

        if self.is_open is False:
            raise ValueError('please open optimizer first!!!')
        else:
            ixs = jnp.arange(self.num_batches)
            bxs = self.x_train.reshape(self.num_batches, self.batch_size, *self.x_train.shape[1:])
            bys = self.y_train.reshape(self.num_batches, self.batch_size, *self.y_train.shape[1:])

            if self.key is not None:
                subkeys = random.split(self.key, self.num_batches + 1)
                self.key, subkeys = subkeys[0], subkeys[1:]  # update self.key & get subkeys

            def one_batch(carry, ix):

                def momentum(d_w, w, v):
                    new_v = self.beta * v + (1 - self.beta) * d_w
                    new_w = w - self.lr * new_v
                    return jnp.stack((new_w, new_v))

                bx = bxs[ix]
                by = bys[ix]

                if self.key is not None:
                    kkey = subkeys[ix]
                    d_params = grad(self._loss, argnums=0)(carry['params'], bx, by, kkey)
                else:
                    d_params = grad(self._loss, argnums=0)(carry['params'], bx, by)

                pack = tree.map(momentum, d_params, carry['params'], carry['V'])
                carry['params'] = tree.map(lambda x: x[0], pack)
                carry['V'] = tree.map(lambda x: x[1], pack)
                carry['steps'] += 1

                return carry, None

            pack, _ = lax.scan(one_batch, {
                'params': self.params,
                'V': self.V,
                'steps': self.steps,
            }, ixs)

            self.params, self.V, self.steps = pack['params'], pack['V'], pack['steps']


class Nesterov(Optimizter):
    '''
    Nesterov Accelerated Gradient (NAG) optimizer, an extension of Momentum that improves
    convergence by incorporating a "lookahead" step. It first makes a momentum-based update
    and then corrects the update using the gradient at the new position.

    Attributes:
        params: The model parameters to be optimized.
        lr: Learning rate (default: 0.01).
        beta: Momentum factor (default: 0.9).
        batch_size: Size of each training batch (default: 32).
        V: Velocity vector (accumulated gradients).
        steps: Number of optimization steps taken so far.
        open: Boolean indicating whether the optimizer is active.
    '''

    def __init__(self, params,
                 lr=0.01, beta=0.9,
                 batch_size=32):
        '''
        Initializes the Nesterov Accelerated Gradient (NAG) optimizer.

        Args:
            params: Model parameters to be optimized.
            lr: Learning rate (default: 0.01).
            beta: Momentum factor (default: 0.9).
            batch_size: Size of each training batch (default: 32).
        '''

        super().__init__()

        self.params = params

        self.lr = lr
        self.beta = beta

        self.batch_size = batch_size

    def flash(self):
        '''
        Resets the optimizer's internal state, including the velocity vector and step count.
        This method is called when the optimizer is opened or re-initialized.
        '''

        self.V = tree.map(lambda x: jnp.zeros_like(x), self.params)

        self.steps = 0
        self.is_open = False

    def update(self):
        '''
        Performs a single optimization step using Nesterov Accelerated Gradient (NAG).
        It updates the model parameters based on the gradients computed from the loss function
        and the accumulated velocity, with a "lookahead" correction.

        Raises:
            ValueError: If the optimizer is not open.
        '''

        if self.is_open is False:
            raise ValueError('please open optimizer first!!!')
        else:
            ixs = jnp.arange(self.num_batches)
            bxs = self.x_train.reshape(self.num_batches, self.batch_size, *self.x_train.shape[1:])
            bys = self.y_train.reshape(self.num_batches, self.batch_size, *self.y_train.shape[1:])

            if self.key is not None:
                subkeys = random.split(self.key, self.num_batches + 1)
                self.key, subkeys = subkeys[0], subkeys[1:]  # update self.key & get subkeys

            def one_batch(carry, ix):

                def nag(d_w, w, v):
                    # Nesterov update
                    v_prev = v
                    v = self.beta * v_prev - self.lr * d_w
                    new_w = w + self.beta * v - self.lr * d_w
                    return jnp.stack((new_w, v))

                bx = bxs[ix]
                by = bys[ix]

                if self.key is not None:
                    kkey = subkeys[ix]
                    d_params = grad(self._loss, argnums=0)(carry['params'], bx, by, kkey)
                else:
                    d_params = grad(self._loss, argnums=0)(carry['params'], bx, by)

                pack = tree.map(nag, d_params, carry['params'], carry['V'])
                carry['params'] = tree.map(lambda x: x[0], pack)
                carry['V'] = tree.map(lambda x: x[1], pack)
                carry['steps'] += 1

                return carry, None

            pack, _ = lax.scan(one_batch, {
                'params': self.params,
                'V': self.V,
                'steps': self.steps,
            }, ixs)

            self.params, self.V, self.steps = pack['params'], pack['V'], pack['steps']


class AdaGrad(Optimizter):
    '''
    AdaGrad (Adaptive Gradient) optimizer, an algorithm that adapts the learning rate for each
    parameter based on the historical gradients. It performs larger updates for infrequent
    parameters and smaller updates for frequent parameters, making it well-suited for sparse data.

    Attributes:
        params: The model parameters to be optimized.
        lr: Learning rate (default: 0.01).
        epsilon: Small constant for numerical stability (default: 1e-8).
        batch_size: Size of each training batch (default: 32).
        G: Accumulated squared gradients for each parameter.
        steps: Number of optimization steps taken so far.
        open: Boolean indicating whether the optimizer is active.
    '''

    def __init__(self, params,
                 lr=0.01, epsilon=1e-8,
                 batch_size=32):
        '''
        Initializes the AdaGrad optimizer.

        Args:
            params: Model parameters to be optimized.
            lr: Learning rate (default: 0.01).
            epsilon: Small constant for numerical stability (default: 1e-8).
            batch_size: Size of each training batch (default: 32).
        '''

        super().__init__()

        self.params = params

        self.lr = lr
        self.epsilon = epsilon

        self.batch_size = batch_size

    def flash(self):
        '''
        Resets the optimizer's internal state, including the accumulated squared gradients
        and step count. This method is called when the optimizer is opened or re-initialized.
        '''

        self.G = tree.map(lambda x: jnp.zeros_like(x), self.params)

        self.steps = 0
        self.is_open = False

    def update(self):
        '''
        Performs a single optimization step using AdaGrad. It updates the model parameters
        based on the gradients computed from the loss function and the accumulated squared gradients.

        Raises:
            ValueError: If the optimizer is not open.
        '''

        if self.is_open is False:
            raise ValueError('please open optimizer first!!!')
        else:
            ixs = jnp.arange(self.num_batches)
            bxs = self.x_train.reshape(self.num_batches, self.batch_size, *self.x_train.shape[1:])
            bys = self.y_train.reshape(self.num_batches, self.batch_size, *self.y_train.shape[1:])

            if self.key is not None:
                subkeys = random.split(self.key, self.num_batches + 1)
                self.key, subkeys = subkeys[0], subkeys[1:]  # update self.key & get subkeys

            def one_batch(carry, ix):

                def adagrad(d_w, w, g):
                    new_g = g + d_w ** 2
                    new_w = w - self.lr * d_w / (jnp.sqrt(new_g) + self.epsilon)
                    return jnp.stack((new_w, new_g))

                bx = bxs[ix]
                by = bys[ix]

                if self.key is not None:
                    kkey = subkeys[ix]
                    d_params = grad(self._loss, argnums=0)(carry['params'], bx, by, kkey)
                else:
                    d_params = grad(self._loss, argnums=0)(carry['params'], bx, by)

                pack = tree.map(adagrad, d_params, carry['params'], carry['G'])
                carry['params'] = tree.map(lambda x: x[0], pack)
                carry['G'] = tree.map(lambda x: x[1], pack)
                carry['steps'] += 1

                return carry, None

            pack, _ = lax.scan(one_batch, {
                'params': self.params,
                'G': self.G,
                'steps': self.steps,
            }, ixs)

            self.params, self.G, self.steps = pack['params'], pack['G'], pack['steps']


class RMSProp(Optimizter):
    '''
    RMSProp (Root Mean Square Propagation) optimizer, an adaptive learning rate optimization
    algorithm that uses a moving average of squared gradients to normalize the gradient updates.
    It addresses the issue of monotonically decreasing learning rates in AdaGrad by using an
    exponentially decaying average of squared gradients.

    Attributes:
        params: The model parameters to be optimized.
        lr: Learning rate (default: 0.01).
        beta: Decay rate for the moving average of squared gradients (default: 0.9).
        epsilon: Small constant for numerical stability (default: 1e-8).
        batch_size: Size of each training batch (default: 32).
        G: Exponentially decaying average of squared gradients for each parameter.
        steps: Number of optimization steps taken so far.
        open: Boolean indicating whether the optimizer is active.
    '''

    def __init__(self, params,
                 lr=0.01, beta=0.9, epsilon=1e-8,
                 batch_size=32):
        '''
        Initializes the RMSProp optimizer.

        Args:
            params: Model parameters to be optimized.
            lr: Learning rate (default: 0.01).
            beta: Decay rate for the moving average of squared gradients (default: 0.9).
            epsilon: Small constant for numerical stability (default: 1e-8).
            batch_size: Size of each training batch (default: 32).
        '''

        super().__init__()

        self.params = params

        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon

        self.batch_size = batch_size

    def flash(self):
        '''
        Resets the optimizer's internal state, including the moving average of squared gradients
        and step count. This method is called when the optimizer is opened or re-initialized.
        '''

        self.G = tree.map(lambda x: jnp.zeros_like(x), self.params)

        self.steps = 0
        self.is_open = False

    def update(self):
        '''
        Performs a single optimization step using RMSProp. It updates the model parameters
        based on the gradients computed from the loss function and the moving average of
        squared gradients.

        Raises:
            ValueError: If the optimizer is not open.
        '''

        if self.is_open is False:
            raise ValueError('please open optimizer first!!!')
        else:
            ixs = jnp.arange(self.num_batches)
            bxs = self.x_train.reshape(self.num_batches, self.batch_size, *self.x_train.shape[1:])
            bys = self.y_train.reshape(self.num_batches, self.batch_size, *self.y_train.shape[1:])

            if self.key is not None:
                subkeys = random.split(self.key, self.num_batches + 1)
                self.key, subkeys = subkeys[0], subkeys[1:]  # update self.key & get subkeys

            def one_batch(carry, ix):

                def rmsprop(d_w, w, g):
                    new_g = self.beta * g + (1 - self.beta) * d_w ** 2
                    new_w = w - self.lr * d_w / (jnp.sqrt(new_g) + self.epsilon)
                    return jnp.stack((new_w, new_g))

                bx = bxs[ix]
                by = bys[ix]

                if self.key is not None:
                    kkey = subkeys[ix]
                    d_params = grad(self._loss, argnums=0)(carry['params'], bx, by, kkey)
                else:
                    d_params = grad(self._loss, argnums=0)(carry['params'], bx, by)

                pack = tree.map(rmsprop, d_params, carry['params'], carry['G'])
                carry['params'] = tree.map(lambda x: x[0], pack)
                carry['G'] = tree.map(lambda x: x[1], pack)
                carry['steps'] += 1

                return carry, None

            pack, _ = lax.scan(one_batch, {
                'params': self.params,
                'G': self.G,
                'steps': self.steps,
            }, ixs)

            self.params, self.G, self.steps = pack['params'], pack['G'], pack['steps']


class AdaDelta(Optimizter):
    '''
    AdaDelta (Adaptive Delta) optimizer, an adaptive learning rate optimization algorithm that
    dynamically adjusts the learning rate based on a moving window of gradient updates. It
    eliminates the need for a manually set learning rate by using a ratio of the root mean
    square (RMS) of parameter updates to the RMS of gradients.

    Attributes:
        params: The model parameters to be optimized.
        rho: Decay rate for the moving averages of squared gradients and updates (default: 0.9).
        epsilon: Small constant for numerical stability (default: 1e-8).
        batch_size: Size of each training batch (default: 32).
        E_g2: Exponentially decaying average of squared gradients.
        E_dx2: Exponentially decaying average of squared parameter updates.
        steps: Number of optimization steps taken so far.
        open: Boolean indicating whether the optimizer is active.
    '''

    def __init__(self, params,
                 rho=0.9, epsilon=1e-8,
                 batch_size=32):
        '''
        Initializes the AdaDelta optimizer.

        Args:
            params: Model parameters to be optimized.
            rho: Decay rate for the moving averages of squared gradients and updates (default: 0.9).
            epsilon: Small constant for numerical stability (default: 1e-8).
            batch_size: Size of each training batch (default: 32).
        '''

        super().__init__()

        self.params = params

        self.rho = rho
        self.epsilon = epsilon

        self.batch_size = batch_size

    def flash(self):
        '''
        Resets the optimizer's internal state, including the moving averages of squared gradients
        and updates, and the step count. This method is called when the optimizer is opened or
        re-initialized.
        '''

        self.E_g2 = tree.map(lambda x: jnp.zeros_like(x), self.params)  # expontential average of squared gradient
        self.E_dx2 = tree.map(lambda x: jnp.zeros_like(x), self.params)  # expontential average of squared update

        self.steps = 0
        self.is_open = False

    def update(self):
        '''
        Performs a single optimization step using AdaDelta. It updates the model parameters
        based on the gradients computed from the loss function and the moving averages of
        squared gradients and updates.

        Raises:
            ValueError: If the optimizer is not open.
        '''

        if self.is_open is False:
            raise ValueError('please open optimizer first!!!')
        else:
            ixs = jnp.arange(self.num_batches)
            bxs = self.x_train.reshape(self.num_batches, self.batch_size, *self.x_train.shape[1:])
            bys = self.y_train.reshape(self.num_batches, self.batch_size, *self.y_train.shape[1:])

            if self.key is not None:
                subkeys = random.split(self.key, self.num_batches + 1)
                self.key, subkeys = subkeys[0], subkeys[1:]  # update self.key & get subkeys

            def one_batch(carry, ix):

                def adadelta(d_w, w, e_g2, e_dx2):
                    new_e_g2 = self.rho * e_g2 + (1 - self.rho) * d_w ** 2  # update squared gradient
                    delta_w = -jnp.sqrt(e_dx2 + self.epsilon) / jnp.sqrt(new_e_g2 + self.epsilon) * d_w
                    new_w = w + delta_w
                    new_e_dx2 = self.rho * e_dx2 + (1 - self.rho) * delta_w ** 2  # update squared update
                    return jnp.stack((new_w, new_e_g2, new_e_dx2))

                bx = bxs[ix]
                by = bys[ix]

                if self.key is not None:
                    kkey = subkeys[ix]
                    d_params = grad(self._loss, argnums=0)(carry['params'], bx, by, kkey)
                else:
                    d_params = grad(self._loss, argnums=0)(carry['params'], bx, by)

                pack = tree.map(adadelta, d_params, carry['params'], carry['E_g2'], carry['E_dx2'])
                carry['params'] = tree.map(lambda x: x[0], pack)
                carry['E_g2'] = tree.map(lambda x: x[1], pack)
                carry['E_dx2'] = tree.map(lambda x: x[2], pack)
                carry['steps'] += 1

                return carry, None

            pack, _ = lax.scan(one_batch, {
                'params': self.params,
                'E_g2': self.E_g2,
                'E_dx2': self.E_dx2,
                'steps': self.steps,
            }, ixs)

            self.params, self.E_g2, self.E_dx2, self.steps = pack['params'], pack['E_g2'], pack['E_dx2'], pack['steps']
