from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, chunk_size=None, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            chunk_size: Size of the action chunks.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data

        if chunk_size is not None and 'actions' in data and 'chunked_actions' not in data:
            actions = data['actions']
            action_dim = actions.shape[-1]
            chunked_actions = np.zeros((len(actions), chunk_size, action_dim), dtype=actions.dtype)
            for offset in range(chunk_size):
                valid_len = len(actions) - offset
                if valid_len <= 0:
                    break
                chunked_actions[:valid_len, offset] = actions[offset:offset + valid_len]
            data['chunked_actions'] = chunked_actions

        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = True  # Whether to additionally return next actions; set outside the class.
        self.history_K = 0 # Number of history actions to return; set outside the class.
        self.sample_k = False # Whether to sample k for FQL; set outside the class.
        self.max_k = None # Max k for FQL; set outside the class.
        self.saturation_ratio = 0.0 # Ratio of saturated k samples; set outside the class.

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

        self.discount = 1.0

    def precompute_mc_returns(self):
        """Pre-compute discounted return-to-go for the entire dataset once."""
        rewards = self._dict['rewards']
        terminals = np.zeros_like(rewards, dtype=bool)
        if hasattr(self, 'initial_locs'):
            end_idxs = self.initial_locs[1:] - 1
            terminals[end_idxs] = True
            terminals[-1] = True
        else:
            terminals[-1] = True

        mc_returns = np.zeros_like(rewards)
        ret = 0.0
        
        for t in reversed(range(len(rewards))):
            if terminals[t]:
                ret = rewards[t]
            else:
                ret = rewards[t] + self.discount * ret
            mc_returns[t] = ret
            
        self._dict['mc_returns'] = mc_returns

    def __getstate__(self):
        state = self.__dict__.copy()
        # Save FrozenDict slots
        state['_dict'] = self._dict
        if hasattr(self, '_hash'):
            state['_hash'] = self._hash
        return state

    def __setstate__(self, state):
        # Restore FrozenDict slots using object.__setattr__ to bypass immutability
        object.__setattr__(self, '_dict', state.pop('_dict'))
        if '_hash' in state:
            object.__setattr__(self, '_hash', state.pop('_hash'))
        self.__dict__.update(state)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
                if i != self.frame_stack - 1:
                    next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
            next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self['next_observations']))

            batch['observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch['next_observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)

            if 'mid_idxs' in batch:
                mid_idxs = batch['mid_idxs']
                mid_initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, mid_idxs, side='right') - 1]
                mid_obs = []
                for i in reversed(range(self.frame_stack)):
                    cur_idxs = np.maximum(mid_idxs - i, mid_initial_state_idxs)
                    mid_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
                batch['mid_observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *mid_obs)

        if self.p_aug is not None:
            # Apply random-crop image augmentation.
            if np.random.rand() < self.p_aug:
                keys = ['observations', 'next_observations']
                if 'mid_observations' in batch:
                    keys.append('mid_observations')
                self.augment(batch, keys)
        return batch

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            next_idxs = np.minimum(idxs + 1, self.size - 1)
            next_actions = self._dict['actions'][next_idxs].copy()

            # Handle terminal states: if current state is terminal, next action is invalid.
            # We replace it with the current action to avoid leaking info from next trajectory.
            if 'terminals' in result:
                mask = result['terminals'].squeeze().astype(bool)
                if np.any(mask):
                    next_actions[mask] = result['actions'][mask]

            result['next_actions'] = next_actions
        
        if self.history_K > 0:
            initial_locs = self.initial_locs
            all_actions = self._dict['actions']
            act_dim = all_actions.shape[-1]
            
            # Find start of current trajectory
            start_idxs_indices = np.searchsorted(initial_locs, idxs, side='right') - 1
            start_idxs = initial_locs[start_idxs_indices]
            
            # History
            history_list = []
            for k in range(self.history_K, 0, -1):
                target_idxs = idxs - k
                valid_mask = target_idxs >= start_idxs
                
                actions_k = np.zeros((len(idxs), act_dim), dtype=all_actions.dtype)
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    valid_target_idxs = target_idxs[valid_indices]
                    actions_k[valid_indices] = all_actions[valid_target_idxs]
                
                history_list.append(actions_k)
            
            result['history'] = np.concatenate(history_list, axis=1)
            
            # Next History
            history_next_list = []
            for k in range(self.history_K, 0, -1):
                target_idxs = idxs - k + 1
                valid_mask = target_idxs >= start_idxs
                
                actions_k = np.zeros((len(idxs), act_dim), dtype=all_actions.dtype)
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    valid_target_idxs = target_idxs[valid_indices]
                    actions_k[valid_indices] = all_actions[valid_target_idxs]
                
                history_next_list.append(actions_k)
            
            result['next_history'] = np.concatenate(history_next_list, axis=1)

        if self.sample_k:
            traj_idxs = np.searchsorted(self.initial_locs, idxs, side='right') - 1
            next_traj_starts = np.zeros_like(idxs)
            not_last_mask = traj_idxs < len(self.initial_locs) - 1
            next_traj_starts[not_last_mask] = self.initial_locs[traj_idxs[not_last_mask] + 1]
            next_traj_starts[~not_last_mask] = self.size
            
            remaining_steps = next_traj_starts - idxs
            
            ks = np.zeros(len(idxs), dtype=np.int32)
            mid_idxs = idxs.copy()
            
            if self.max_k is not None:
                # Sample ks from [0, max_k]
                rem = np.maximum(remaining_steps, 1)
                
                # Saturated range: [ceil(log2(rem + 0.5)), max_k]
                min_ks_sat = np.ceil(np.log2(rem + 0.5)).astype(np.int32)
                # Non-saturated range: [0, floor(log2(rem))]
                max_ks_non_sat = np.floor(np.log2(rem)).astype(np.int32)
                
                is_sat_sample = np.random.rand(len(idxs)) < self.saturation_ratio
                # Ensure we can actually sample a saturated k
                is_sat_sample &= (min_ks_sat <= self.max_k)
                
                for i in range(len(idxs)):
                    if is_sat_sample[i]:
                        ks[i] = np.random.randint(min_ks_sat[i], self.max_k + 1)
                    else:
                        ks[i] = np.random.randint(0, min(max_ks_non_sat[i], self.max_k) + 1)
                
                ds = 2 ** ks
                is_saturated = ds > remaining_steps
                valid_mid_mask = (~is_saturated) & (ks > 0)
                mid_idxs[valid_mid_mask] = idxs[valid_mid_mask] + (2 ** (ks[valid_mid_mask] - 1))
            else:
                valid_mask = remaining_steps > 2
                if np.any(valid_mask):
                    max_ks = np.floor(np.log2(remaining_steps[valid_mask] - 1)).astype(np.int32)
                    max_ks = np.maximum(max_ks, 1)
                    sampled_ks = np.random.randint(1, max_ks + 1)
                    ks[valid_mask] = sampled_ks
                    
                    ds = 2 ** sampled_ks
                    mid_idxs[valid_mask] = idxs[valid_mask] + ds // 2
            
            result['ks'] = ks
            result['mid_idxs'] = mid_idxs
            result['mid_observations'] = self._dict['observations'][mid_idxs]
            result['mid_actions'] = self._dict['actions'][mid_idxs]
            result['remaining_steps'] = remaining_steps
            
            # get mc returns for remaining steps
            if not 'mc_returns' in self._dict:
                self.precompute_mc_returns()
            result['mc_returns'] = self._dict['mc_returns'][idxs]

        # print(self.return_next_actions, result.keys())
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )

    def compute_statistics(self):
        """Compute statistics of the dataset."""
        stats = {}
        if 'observations' in self._dict:
            stats['observations_min'] = np.min(self._dict['observations'], axis=0)
            stats['observations_max'] = np.max(self._dict['observations'], axis=0)
        if 'actions' in self._dict:
            stats['actions_min'] = np.min(self._dict['actions'], axis=0)
            stats['actions_max'] = np.max(self._dict['actions'], axis=0)
        if 'rewards' in self._dict:
            stats['rewards_min'] = np.min(self._dict['rewards'], axis=0)
            stats['rewards_max'] = np.max(self._dict['rewards'], axis=0)
        return stats


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""
        for key in self._dict.keys():
            if key in transition:
                self._dict[key][self.pointer] = transition[key]
            else:
                self._dict[key][self.pointer] = 0

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0
