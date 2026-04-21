import copy
import functools
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value, CriticVectorField, CriticResVectorField
import flax.linen as nn

class Scalar(nn.Module):
    init_value: float
    @nn.compact
    def __call__(self):
        log_alpha = self.param('value', lambda rng: jnp.array([self.init_value]))
        return log_alpha

class DeFlowVFAgent(flax.struct.PyTreeNode):
    """DeFlow with Floq (using flow matching to model value)"""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch,  grad_params, rng,):
        """Compute the FQL critic loss."""

        def repeat_tensor(x, n): #(b,..) -> (b*n,...)
            batch_size, non_batch_dims = x.shape[0], x.shape[1:]
            return x[:, None].repeat(n, axis = 1).reshape(batch_size*n, *non_batch_dims)

        batch_size, observation_dim = batch['observations'].shape #(b,o)
        repeat = self.config['noise_samples'] #r

        observations = repeat_tensor(batch['observations'], repeat) #(b*r, o)
        actions = repeat_tensor(batch['actions'], repeat) #(b*r, a)
        next_observations = repeat_tensor(batch['next_observations'], repeat) #(b*r, o)

        #sample actions
        rng, sample_rng = jax.random.split(rng, 2)
        next_actions = jnp.clip(self.sample_actions(next_observations, seed=sample_rng,), -1, 1) #(b*r, a)

        #compute target returns (used for TD backup)
        rng, next_noise_rng = jax.random.split(rng)
        next_noise_ratios = jax.random.uniform(next_noise_rng, (batch_size*repeat, 1)) #(b*r, 1)
        next_returns = self.compute_target_flow_returns(next_observations, 
                                                        next_actions, 
                                                        noise_ratios = next_noise_ratios,).squeeze(-1).reshape((batch_size, repeat)) #(b, r)
        
        rewards, masks = batch['rewards'].reshape((batch_size, 1)), batch['masks'].reshape((batch_size, 1)) #(b, 1), #(b, 1) 
        target_returns = (rewards + self.config['discount'] * masks * next_returns).mean(axis = -1) #(b,r)

        assert target_returns.shape == (batch_size,)

        #compute current returns (used for distillation)
        rng, noise_rng =  jax.random.split(rng) 
        noise_ratios = jax.random.uniform(noise_rng, (batch_size*repeat, 1)) #(b*r, 1)
        current_returns = self.compute_current_flow_returns(observations, 
                                                            actions, 
                                                            noise_ratios = noise_ratios,) #(e, b*r, 1) where e=num_ensembles

        current_returns = current_returns.reshape((self.config['flow_num_ensembles'], batch_size, repeat)) #(e, b, r)

        #compute losses -- flow critic loss
        x_0 =  (1. - noise_ratios)*self.config['noise_min'] + noise_ratios*self.config['noise_max'] #(b*r,1) 
        
        x_1 = jnp.tile(target_returns.reshape((1, batch_size, 1)), [self.config['flow_num_ensembles'], 1, repeat]) #(e, b, r)

        x_1 = x_1.reshape((self.config['flow_num_ensembles'], batch_size*repeat, 1)) #(e, b*r, 1)

        rng, t_rng = jax.random.split(rng, 2)

        if self.config['train_at_zero_only']:
            t = jnp.zeros((batch_size*repeat, 1)) #(b*r, 1)
        else:
            t = jax.random.uniform(t_rng, (batch_size*repeat, 1)) #(b*r, 1)

        x_t = (jnp.expand_dims(1. - t, axis = 0)) * jnp.expand_dims(x_0, axis = 0) + jnp.expand_dims(t, axis = 0) * x_1 #(e, b*r, 1)
        assert x_t.ndim == 3 and x_t.shape[0] == self.config['flow_num_ensembles']
        vel = x_1 - jnp.expand_dims(x_0, axis = 0) #(e, b*r,1) 

        pred = self.network.select('floq')(observations, actions, returns = x_t, times = t, params = grad_params) #(e, b*r, 1)
        vel = vel.reshape((self.config['flow_num_ensembles'], batch_size, repeat)) #(e,b,r)
        pred = pred.reshape((self.config['flow_num_ensembles'], batch_size, repeat)) #(e,b,r)

        floq_loss = jnp.sum((pred -  vel)**2, axis = -1) #(e,b),  sum loss over initial noises

        # compute losses -- distill critic loss
        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params) #(e,b)

        assert current_returns.shape == (self.config['flow_num_ensembles'], batch_size, repeat)
        mean_current_returns = current_returns.mean(axis = -1).mean(axis = 0) #(b,)

        distilled_critic_loss = jnp.square(q - mean_current_returns) #(e,b)
        assert distilled_critic_loss.shape == floq_loss.shape
        critic_loss = floq_loss + distilled_critic_loss 

        #metrics for logging purposes
        current_returns = jnp.mean(current_returns, axis = 0) #(batch_size, repeat)
        var_flow_returns = jnp.var(current_returns, axis = -1) #(batch_size,)
        td_error = (mean_current_returns - target_returns)**2

        return critic_loss.mean(), {
            'floq_loss': floq_loss.mean(), 
            'distilled_critic_loss': distilled_critic_loss.mean(),
            'critic_loss': critic_loss.mean(),
            'q': q.mean(),
            'var_flow_returns': jnp.mean(var_flow_returns),
            'td_error':  td_error.mean(),
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
    def update(self, batch,):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng,)
        

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'floq')

        return self.replace(network=new_network, rng=new_rng), info

    def compute_target_flow_returns(
        self,
        observations,
        actions,
        noise_ratios,
    ):
        """Compute returns from the critic flow model using the Euler method."""
        
        assert observations.ndim == 2 and actions.ndim == 2 #(batch_size, observation_dim), (batch_size, action_dim)
        assert noise_ratios.ndim == 2 and noise_ratios.shape[-1] == 1 #(batch_size, 1) because return is 1-d.

        if self.config['encoder'] is not None:
            observations = self.network.select('target_floq_encoder')(observations) #(batch_size, encoding_dim)

        returns = self.config['noise_min']*(1 - noise_ratios) + self.config['noise_max']*noise_ratios #(batch_size, 1)
        assert returns.ndim == 2
        returns = jnp.expand_dims(returns, axis = 0) #(1, batch_size, 1)
        returns = jnp.tile(returns, [self.config['flow_num_ensembles'], 1, 1]) #(flow_num_ensembles, batch_size, 1)

        for i in range(self.config['critic_flow_steps']):    

            t = jnp.full((*observations.shape[:-1], 1), i / self.config['critic_flow_steps']) #(batch_size, 1)
            vels = self.network.select('target_floq')(observations, actions, returns = returns, times = t,) #(flow_num_ensembles, batch_size, 1)
            assert vels.ndim  == 3 and vels.shape[0] == self.config['flow_num_ensembles']
            returns = returns + vels/ self.config['critic_flow_steps'] #(flow_num_ensembles, batch_size, 1)

        if self.config['q_agg'] == 'min':
            returns = jnp.min(returns, axis = 0) #(batch_size, 1)
        else:
            returns = jnp.mean(returns, axis = 0) #(batch_size, 1)

        return returns # (batch_size, 1)

    @jax.jit
    def compute_current_flow_returns(
        self,
        observations,
        actions,
        noise_ratios,
    ):
        """Compute returns from the critic flow model using the Euler method."""

        assert observations.ndim == 2 and actions.ndim == 2 #(batch_size, observation_dim), (batch_size, action_dim)
        assert noise_ratios.ndim == 2 and noise_ratios.shape[-1] == 1 #(batch_size, 1) because return is 1-d.

        if self.config['encoder'] is not None:
            observations = self.network.select('target_floq_encoder')(observations) #(batch_size, encoding_dim)

        returns = self.config['noise_min']*(1 - noise_ratios) + self.config['noise_max']*noise_ratios #(batch_size, 1)
        assert returns.ndim == 2
        returns = jnp.expand_dims(returns, axis = 0) #(1, batch_size, 1)
        returns = jnp.tile(returns, [self.config['flow_num_ensembles'], 1, 1]) #(flow_num_ensembles, batch_size, 1)

        for i in range(self.config['critic_flow_steps']):    

            t = jnp.full((*observations.shape[:-1], 1), i / self.config['critic_flow_steps']) #(batch_size, 1)
            vels = self.network.select('floq')(observations, actions, returns = returns, times = t,) #(flow_num_ensembles, batch_size, 1)
            assert vels.ndim  == 3 and vels.shape[0] == self.config['flow_num_ensembles']
            returns = returns + vels/ self.config['critic_flow_steps'] #(flow_num_ensembles, batch_size, 1)

        return returns #(flow_num_ensembles, batch_size, 1)

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
        critic_ex_times = ex_actions[..., :1] #(batch_size, 1)
        actor_ex_times = ex_actions[..., :1] #(batch_size, 1)
        ex_returns = ex_actions[..., :1] #(batch_size, 1)
        ex_returns = jnp.tile(jnp.expand_dims(ex_returns, axis = 0), [config['flow_num_ensembles'], 1, 1]) #(flow_num_ensembles, batch_size, 1)
        batch_size = ex_returns.shape[0]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['refine_onestep_flow'] = encoder_module()
            encoders['floq'] = encoder_module()

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['distill_num_ensembles'],
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

        q_min = (config['r_min'] - config['reward_offset'])/(1. - config['discount'])
        q_max = (config['r_max'] + config['reward_offset'])/(1. - config['discount'])

        if config['use_resnets']:   
            per_transformation_hidden_dims = config['block_depth']*[config['block_width'],]
            floq_def = CriticResVectorField(
                num_transformations=config['num_transformations'],
                per_transformation_hidden_dims=per_transformation_hidden_dims,
                activate_before_layernorm= config['activate_before_layernorm'],
                layer_norm=config['layer_norm'],
                encoder = encoders.get('floq'),
                num_ensembles = config['flow_num_ensembles'],
                embed_time=config['embed_time'],
                time_embed_dim=config['time_embed_dim'],
                use_prob_embed = config['use_prob_embed'],
                q_min = q_min,
                q_max = q_max,
                num_bins = config['num_bins'],
                sigma = config['sigma'],
            ) 
        else:
            hidden_dims = config['block_depth']*[config['block_width']]
            floq_def = CriticVectorField(
                hidden_dims=hidden_dims,
                layer_norm=config['layer_norm'],
                encoder = encoders.get('floq'),
                num_ensembles = config['flow_num_ensembles'],
                embed_time=config['embed_time'],
                time_embed_dim=config['time_embed_dim'],
                use_prob_embed = config['use_prob_embed'],
                q_min = q_min,
                q_max = q_max,
                num_bins = config['num_bins'],
                sigma = config['sigma'],
            ) 


        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions,)),
            floq = (floq_def, (ex_observations, ex_actions, ex_returns,  critic_ex_times,)),            
            target_floq = (copy.deepcopy(floq_def), (ex_observations, ex_actions, ex_returns, critic_ex_times,)),
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
        if encoders.get('floq') is not None:
            network_info['floq_encoder'] = (encoders.get('floq'), (ex_observations,))
            network_info['target_floq_encoder']  = copy.deepcopy(network_info['floq_encoder'])
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']

        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_floq'] = params['modules_floq']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        config['noise_min'] = config['noise_coverage']*(config['r_min']/(1. - config['discount']))
        config['noise_max'] = config['noise_coverage']*(config['r_max']/(1. - config['discount'])) #usually zero in all envs we consider

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='deflowvf',  # Agent name.
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
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            use_lagrange=True,     # Enable automatic tuning
            target_divergence=0.001, # Target MSE for delta (approx delta ~ 0.22)
            fix_bc_flow_online=False,  # Whether to fix the BC flow during online fine-tuning.
            offline_steps=1000000,  # Number of offline steps (will be set automatically).
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            flow_num_ensembles=2, #Number of flow critic ensembles.
            distill_num_ensembles=2, #Number of distill critic ensembles.
            noise_samples=8, #Number of noise samples used to compute Q-values.
            noise_coverage = 0.1, #Defines range of initial noise w.r.t range of Q-values.
            critic_flow_steps= 8,  #Number of critic flow steps.
            train_at_zero_only = False, #Whether to train a one-step flow, set critic_flow_steps=1 if using this flag. 
            embed_time=True, #Whether to use fourier time embedding.
            time_embed_dim=64, #Dimension of fourier time embedding.
            use_prob_embed = True, #Whether to use HL-Gauss interpolant embedding.
            num_bins = 51, #Number of bins for HL-Gauss interpolant embedding.
            sigma = 16.0, #HL-Gauss sigma for interpolant embedding.
            reward_offset = 0.01, #Define range of HL-Gauss embed bins which is set slightly wider than Q-value range to prevent edge effects.
            block_width=512, #Width of flow velocity network.
            block_depth=4, #Depth of flow velocity network (depth of each block if resnet).
            use_resnets=False, #Whether to use resnet flow velocity network.
            num_transformations=2, #Number of residual blocks of size block_depth*[block_width,] if use_resnets. 
            activate_before_layernorm=True, #Whether to have activation before layernorm, we found this works better than the other way round.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config