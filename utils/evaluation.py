from collections import defaultdict

import jax
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=1,
    external_agent=None,
    scaling_method='default',
    num_samples=10,
    num_refine_steps=5,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        external_agent: Optional external agent to provide soft evidence.
        scaling_method: Scaling method (default, max_q, refinement).
        num_samples: Number of samples for max_q scaling.
        num_refine_steps: Number of refinement steps for refinement scaling.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    if hasattr(agent, 'pc'): # Check if it's PCAgent (PyTorch)
        # PCAgent uses torch, doesn't need JAX RNG key splitting
        def actor_fn(observations, temperature=1.0, seed=None, external_logps=None):
            # PCAgent.sample_actions expects observations, seed, temperature
            # We can just pass seed if provided, or let it handle randomness
            return agent.sample_actions(observations, seed=seed, temperature=temperature, external_logps=external_logps)
    else:
        # JAX Agent
        if scaling_method == 'max_q' and hasattr(agent, 'sample_actions_max_q'):
            def sample_fn(*args, **kwargs):
                return agent.sample_actions_max_q(*args, num_samples=num_samples, **kwargs)
            actor_fn = supply_rng(sample_fn, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
        elif scaling_method == 'refinement' and hasattr(agent, 'sample_actions_iterative_refinement'):
            def sample_fn(*args, **kwargs):
                return agent.sample_actions_iterative_refinement(*args, num_refine_steps=num_refine_steps, **kwargs)
            actor_fn = supply_rng(sample_fn, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
        else:
            actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
        
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        if hasattr(agent, 'reset'):
            agent.reset()

        observation, info = env.reset()
        done = False
        step = 0
        render = []
        while not done:
            if hasattr(agent, 'pc'):
                # PCAgent returns torch tensor, need to convert to numpy
                ext_lls = None
                if external_agent is not None:
                    # Ensure observation has batch dim for external agent
                    obs_batch = observation[None, ...]
                    logits = external_agent.get_action_logits(obs_batch)
                    ext_lls = agent.compute_external_evidence(logits)

                action = actor_fn(observations=observation, temperature=eval_temperature, external_logps=ext_lls)
                if hasattr(action, 'cpu'):
                    action = action.cpu().numpy()
                # If it's a batch of 1, squeeze it? 
                # sample_actions usually returns [B, D]. Here observation is [D] or [1, D]?
                # env.reset() returns [D].
                # If we passed [D] to sample_actions, it might return [D] or [1, D].
                # Let's ensure it's flat for env.step
                if len(action.shape) > 1:
                    action = action.squeeze(0)
            else:
                action = actor_fn(observations=observation, temperature=eval_temperature)
                
            action = np.array(action)
            action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
