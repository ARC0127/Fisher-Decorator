import functools
import glob
import os
import pickle
from typing import Any, Dict, Mapping, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class ModuleDict(nn.Module):
    """A dictionary of modules.

    This allows sharing parameters between modules and provides a convenient way to access them.

    Attributes:
        modules: Dictionary of modules.
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f'When `name` is not specified, kwargs must contain the arguments for each module. '
                    f'Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}'
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Custom train state for models.

    Attributes:
        step: Counter to keep track of the training steps. It is incremented by 1 after each `apply_gradients` call.
        apply_fn: Apply function of the model.
        model_def: Model definition.
        params: Parameters of the model.
        tx: optax optimizer.
        opt_state: Optimizer state.
    """

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any
    # Optional centralized grad clipping settings
    grad_clip: Any = nonpytree_field(default=None)  # float or None
    grad_clip_include: Any = nonpytree_field(default=None)  # list/tuple of substrings to include for mask (clip only these)
    grad_clip_exclude: Any = nonpytree_field(default=None)  # list/tuple of substrings to exclude from mask

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        """Create a new train state."""
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        """Forward pass.

        When `params` is not provided, it uses the stored parameters.

        The typical use case is to set `params` to `None` when you want to *stop* the gradients, and to pass the current
        traced parameters when you want to flow the gradients. In other words, the default behavior is to stop the
        gradients, and you need to explicitly provide the parameters to flow the gradients.

        Args:
            *args: Arguments to pass to the model.
            params: Parameters to use for the forward pass. If `None`, it uses the stored parameters, without flowing
                the gradients.
            method: Method to call in the model. If `None`, it uses the default `apply` method.
            **kwargs: Keyword arguments to pass to the model.
        """
        if params is None:
            params = self.params
        variables = {'params': params}
        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None

        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        """Helper function to select a module from a `ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        """Apply the gradients and return the updated state."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn):
        """Apply the loss function and return the updated state and info.

        It additionally computes the gradient statistics and adds them to the dictionary.
        """
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)

        grad_max = jax.tree_util.tree_map(jnp.max, grads)
        grad_min = jax.tree_util.tree_map(jnp.min, grads)
        grad_norm = jax.tree_util.tree_map(jnp.linalg.norm, grads)

        grad_max_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_max)], axis=0)
        grad_min_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_min)], axis=0)
        grad_norm_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_norm)], axis=0)

        final_grad_max = jnp.max(grad_max_flat)
        final_grad_min = jnp.min(grad_min_flat)
        final_grad_norm = jnp.linalg.norm(grad_norm_flat, ord=1)

        info.update({
            'grad/max': final_grad_max,
            'grad/min': final_grad_min,
            'grad/norm': final_grad_norm,
        })

        # Centralized gradient clipping (optional). Supports masking by param path substrings.
        clipped_grads = grads
        clip_scale = None
        if self.grad_clip is not None:
            # Build mask from include/exclude substrings.
            include = tuple(self.grad_clip_include) if self.grad_clip_include else None
            exclude = tuple(self.grad_clip_exclude) if self.grad_clip_exclude else None

            def build_mask(tree, path=()):
                if isinstance(tree, dict):
                    return {k: build_mask(v, path + (k,)) for k, v in tree.items()}
                # Leaf: decide mask by path string
                path_str = '/'.join(path)
                inc_ok = True if include is None else any(s in path_str for s in include)
                exc_ok = True if exclude is None else not any(s in path_str for s in exclude)
                return bool(inc_ok and exc_ok)

            if include is None and exclude is None:
                # Clip all gradients by global norm
                global_norm = optax.global_norm(clipped_grads)
                # scale = min(1, clip / (norm + 1e-6))
                clip_scale = jnp.minimum(1.0, self.grad_clip / (global_norm + 1e-6))
                clipped_grads = jax.tree_util.tree_map(lambda g: g * clip_scale, clipped_grads)
            else:
                mask_tree = build_mask(self.params)
                # Split grads into masked and rest
                def masked_part(g, m):
                    return jnp.where(m, g, jnp.zeros_like(g))
                def rest_part(g, m):
                    return jnp.where(m, jnp.zeros_like(g), g)
                g_masked = jax.tree_util.tree_map(masked_part, clipped_grads, mask_tree)
                g_rest = jax.tree_util.tree_map(rest_part, clipped_grads, mask_tree)
                # Compute norm only over masked subset
                norm_masked = optax.global_norm(g_masked)
                clip_scale = jnp.minimum(1.0, self.grad_clip / (norm_masked + 1e-6))
                g_masked = jax.tree_util.tree_map(lambda g: g * clip_scale, g_masked)
                # Merge back
                clipped_grads = jax.tree_util.tree_map(lambda a, b: a + b, g_masked, g_rest)

            info['grad/clip_scale'] = clip_scale if clip_scale is not None else jnp.array(1.0)
            # Optional: log clipped grad global norm
            info['grad/norm_clipped'] = optax.global_norm(clipped_grads)

        return self.apply_gradients(grads=clipped_grads), info


def save_agent(agent, save_dir, epoch):
    """Save the agent to a file.

    Args:
        agent: Agent.
        save_dir: Directory to save the agent.
        epoch: Epoch number.
    """

    save_dict = dict(
        agent=flax.serialization.to_state_dict(agent),
    )
    save_path = os.path.join(save_dir, f'params_{epoch}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f'Saved to {save_path}')


def restore_agent(agent, restore_path, restore_epoch):
    """Restore the agent from a file.

    Args:
        agent: Agent.
        restore_path: Path to the directory containing the saved agent.
        restore_epoch: Epoch number.
    """
    candidates = glob.glob(restore_path)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    restore_path = candidates[0] + f'/params_{restore_epoch}.pkl'

    with open(restore_path, 'rb') as f:
        load_dict = pickle.load(f)

    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])

    print(f'Restored from {restore_path}')

    return agent
