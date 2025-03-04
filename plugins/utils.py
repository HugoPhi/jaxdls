import jax

# L1正则化项
def l1_regularization(params, lambda_l1):
    return lambda_l1 * sum(jnp.abs(p).sum() for p in jax.tree_util.tree_leaves(params))

# L2正则化项
def l2_regularization(params, lambda_l2):
    return lambda_l2 * sum((p ** 2).sum() for p in jax.tree_util.tree_leaves(params))