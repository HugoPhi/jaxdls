import jax
import jax.numpy as jnp
from jax import jit


@jit
def l1_regularization(params, lambda_l1=0.01):  # L1正则化项
    '''
    Computes the L1 regularization term for the given model parameters.

    L1 regularization encourages sparsity in the model parameters by penalizing the sum of their absolute values.

    Args:
        params: A nested structure (e.g., dictionary or list) of model parameters.
        lambda_l1: Regularization strength (default: 0.01).

    Returns:
        The L1 regularization term as a scalar value.
    '''

    return lambda_l1 * sum(jnp.abs(p).sum() for p in jax.tree_util.tree_leaves(params))


@jit
def l2_regularization(params, lambda_l2=0.01):  # L2正则化项
    '''
    Computes the L2 regularization term for the given model parameters.

    L2 regularization discourages large parameter values by penalizing the sum of their squares.

    Args:
        params: A nested structure (e.g., dictionary or list) of model parameters.
        lambda_l2: Regularization strength (default: 0.01).

    Returns:
        The L2 regularization term as a scalar value.
    '''

    return lambda_l2 * sum((p ** 2).sum() for p in jax.tree_util.tree_leaves(params))


@jit
def softmax(logits):
    '''
    Computes the softmax function for the given logits.

    The softmax function converts logits into probabilities by exponentiating and normalizing them.

    Args:
        logits: A 2D array of shape (batch_size, num_classes) containing the raw model outputs.

    Returns:
        A 2D array of shape (batch_size, num_classes) containing the probabilities.
    '''

    logits_stable = logits - jnp.max(logits, axis=1, keepdims=True)
    exp_logits = jnp.exp(logits_stable)
    return exp_logits / jnp.sum(exp_logits, axis=1, keepdims=True)


@jit
def cross_entropy_loss(y, y_pred):
    '''
    Computes the cross-entropy loss between the true labels and predicted probabilities.

    Cross-entropy loss measures the difference between the true label distribution and the predicted
    probability distribution. It is commonly used for classification tasks.

    Args:
        y: A 2D array of shape (batch_size, num_classes) containing the true one-hot encoded labels.
        y_pred: A 2D array of shape (batch_size, num_classes) containing the predicted probabilities.

    Returns:
        The average cross-entropy loss as a scalar value.
    '''

    epsilon = 1e-9  # Small constant to avoid log(0)
    y_pred_clipped = jnp.clip(y_pred, epsilon, 1. - epsilon)  # clip here is very important, or you will get Nan when you training.
    loss = -jnp.sum(y * jnp.log(y_pred_clipped), axis=1)
    return loss.mean()


@jit
def mean_squre_error(y, y_pred):
    '''
    Computes the mean squared error (MSE) between the true values and predicted values.

    MSE measures the average squared difference between the true and predicted values. It is commonly
    used for regression tasks.

    Args:
        y: A 1D or 2D array containing the true values.
        y_pred: A 1D or 2D array containing the predicted values.

    Returns:
        The mean squared error as a scalar value.
    '''

    return jnp.mean((y - y_pred)**2)


@jit
def relu(x: jnp.ndarray):
    '''
    Applies the Rectified Linear Unit (ReLU) activation function to the input.

    ReLU is defined as `max(0, x)`, which sets all negative values to zero and leaves positive values unchanged.

    Args:
        x: A JAX array (1D, 2D, or ND) containing the input values.

    Returns:
        A JAX array of the same shape as `x` with ReLU applied
    '''

    return jnp.maximum(x)


def one_hot(x: jnp.ndarray, num_class):
    '''
    Converts a 1D array of class indices into a 2D one-hot encoded array.

    One-hot encoding represents each class index as a binary vector where the index corresponding to the
    class is set to 1, and all other indices are set to 0.

    Args:
        x: A 1D array of shape (batch_size,) containing class indices.
        num_class: The total number of classes.

    Returns:
        A 2D array of shape (batch_size, num_class) containing the one-hot encoded vectors
    '''

    res = jnp.zeros((x.shape[0], num_class))
    return res.at[jnp.arange(x.shape[0]), x].set(1)


@jit
def sigmoid(x: jnp.ndarray, clip=50):
    '''
    Applies the Sigmoid activation function to the input.

    Sigmoid is defined as `1 / (1 + exp(-x))`, which transforms values into a range between 0 and 1.

    Args:
        x: A JAX array (1D, 2D, or ND) containing the input values.
        clip: The maximum value to clip the input before applying the sigmoid function.
    '''
    x = jnp.clip(x, -clip, clip)
    return 1 / (1 + jnp.exp(-x))
