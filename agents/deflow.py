import copy
import functools
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value
import flax.linen as nn

class Scalar(nn.Module):
    init_value: float
    @nn.compact
    def __call__(self):
        log_alpha = self.param('value', lambda rng: jnp.array([self.init_value]))
        return log_alpha

class DeFlowAgent(flax.struct.PyTreeNode):
    """DeFlow agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        if self.config.get('fix_bc_flow_online', False):
            online = self.network.step > self.config.get('offline_steps', 0)
            bc_flow_loss = jax.lax.cond(online, lambda: 0.0, lambda: bc_flow_loss)

        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        base_actions = self.compute_flow_actions(batch['observations'], noises=noises)

        # stop gradient of base_actions
        base_actions = jax.lax.stop_gradient(base_actions)

        refinement = self.network.select('refine_onestep_flow')(batch['observations'], base_actions, params=grad_params)
        raw_distill_loss = jnp.mean((refinement) ** 2)

        if self.config['use_lagrange']:
            log_alpha = self.network.select('log_alpha')(params=grad_params)[0]
            alpha = jnp.exp(log_alpha)

            # clip alpha in case too large/small
            # alpha = jnp.clip(alpha, 0.01, 100)

            diff = jax.lax.stop_gradient(raw_distill_loss) - self.config['target_divergence']
            alpha_loss = -(log_alpha * diff).sum()
            
            # 在 Actor Loss 中使用当前的 alpha (stop_gradient 避免 actor 更新 alpha)
            distill_loss = jax.lax.stop_gradient(alpha) * raw_distill_loss
        else:
            # Fixed Alpha
            alpha = self.config['alpha']
            alpha_loss = 0.0
            distill_loss = alpha * raw_distill_loss

        actor_actions = base_actions + refinement

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        # Total loss.
        if self.config['use_lagrange']:
            actor_loss = bc_flow_loss + distill_loss + q_loss + alpha_loss
        else:
            actor_loss = bc_flow_loss + distill_loss + q_loss

        # Additional metrics for logging.
        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': raw_distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'alpha_loss': alpha_loss,
            'alpha_value': alpha,
            'mse': mse,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @functools.partial(jax.jit, static_argnames=('num_samples',))
    def sample_actions_max_q(
        self,
        observations,
        seed=None,
        num_samples=10,
        temperature=1.0
    ):
        """Sample multiple actions and select the one with the highest Q-value."""
        obs_repeated = jnp.repeat(observations[None], num_samples, axis=0)
        
        # Sample base actions for all repeated observations
        noises = jax.random.normal(seed, (num_samples, self.config['action_dim']))
        base_actions = self.compute_flow_actions(obs_repeated, noises=noises)
        
        # Compute Q-values for BASE actions
        qs = self.network.select('critic')(obs_repeated, actions=base_actions)
        # Use min across ensembles. qs shape: (num_ensembles, num_samples)
        q_values = qs.mean(axis=0) # (num_samples,)

        best_idx = jnp.argmax(q_values)
        best_base_action = base_actions[best_idx]

        # Refine only the best base action
        refinement = self.network.select('refine_onestep_flow')(observations, best_base_action)
        action = best_base_action + refinement
        action = jnp.clip(action, -1, 1)
        
        # Return best action
        return action

    @functools.partial(jax.jit, static_argnames=('num_refine_steps',))
    def sample_actions_iterative_refinement(
        self,
        observations,
        seed=None,
        num_refine_steps=5,
        temperature=1.0,
    ):
        """Sample actions with iterative refinement."""
        action_seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        
        actions = self.compute_flow_actions(observations, noises=noises)
        
        action_list = [actions]

        # Iterative refinement
        def refine_step(curr_list, _):
            curr_actions = curr_list[-1]
            refinement = self.network.select('refine_onestep_flow')(observations, curr_actions)
            next_actions = curr_actions + refinement
            next_actions = jnp.clip(next_actions, -1, 1)
            curr_list.append(next_actions)
            return curr_list, None
        for i in range(num_refine_steps):
            action_list, _ = refine_step(action_list, None)
        
        # make action_lists to a (num_steps+1, action_dim) jax array
        actions = jnp.stack(action_list, axis=0)

        # repeat observations to match actions shape
        observations = jnp.repeat(observations[None], num_refine_steps + 1, axis=0)

        # calculate q values for all refined actions
        qs = self.network.select('critic')(observations, actions=actions)
        q_values = qs.mean(axis=0)  # (num_steps+1,)

        # select action with highest q value
        best_idx = jnp.argmax(q_values)
        actions = actions[best_idx]

        return actions

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the one-step policy."""
        action_seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        # actions = self.network.select('refine_onestep_flow')(observations, noises)
        actions = self.compute_flow_actions(observations, noises=noises)
        refinement = self.network.select('refine_onestep_flow')(observations, actions)
        actions = actions + refinement
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['refine_onestep_flow'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        refine_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('refine_onestep_flow'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
            refine_onestep_flow=(refine_onestep_flow_def, (ex_observations, ex_actions)),
        )
        # --- Inject log_alpha into params if using Lagrange ---
        if config['use_lagrange']:
            # init_value=2.3 对应 alpha ≈ 10.0
            # 第二个参数 () 是输入的 args，Scalar 不需要输入
            network_info['log_alpha'] = (Scalar(init_value=float(jnp.log(config['alpha']))), ())
        # ----------------------------------------------------
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']

        network = TrainState.create(network_def, network_params, tx=network_tx, grad_clip=config.get('grad_clip', None))

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='deflow',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            grad_clip=5.0,  # FQL Appendix default gradient clipping threshold.
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            use_lagrange=True,     # Enable automatic tuning
            target_divergence=0.001, # Target MSE for delta (approx delta ~ 0.22)
            fix_bc_flow_online=False,  # Whether to fix the BC flow during online fine-tuning.
            offline_steps=1000000,  # Number of offline steps (will be set automatically).
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
