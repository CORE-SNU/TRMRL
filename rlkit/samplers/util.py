import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu


def rollout(env, agent, max_path_length=np.inf, accum_context=True, animated=False, save_frames=False):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param accum_context: if True, accumulate the collected context
    :param animated:
    :param save_frames: if True, save video of rollout
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0

    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        # update the agent's current context
        if accum_context:
            agent.update_context([o, a, r, next_o, d, env_info])
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        path_length += 1
        o = next_o
        if animated:
            pass
            # image = env.render()
        if save_frames:
            # from PIL import Image
            # image = Image.fromarray(np.flipud(env.get_image()))
            image = env.render()
            env_info['frame'] = image
        env_infos.append(env_info)
        if d:
            break

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def model_rollout(model, agent, initial_obs: np.ndarray, max_path_length=np.inf, accum_context=True):

    class LearnedEnv:
        """
        A simple implementation of the virtual system which is simulated by the learned model
        """
        def __init__(self, z: torch.tensor):
            self._obs = None      # for simulated states
            # set T_z(s'|s, a) & R_z(s, a) according to a given latent vector z
            # used as an input of the model when step() is called.
            self._z = z
            return

        def reset(self) -> np.ndarray:
            self._obs = initial_obs
            return np.copy(self._obs)

        def step(self, a: np.ndarray):
            """
            A single step simulation according to the model.
            """

            next_obs, reward = model(ptu.from_numpy(self._obs), ptu.from_numpy(a), torch.squeeze(self._z))

            next_obs = next_obs.detach().cpu().numpy()
            reward = reward.detach().cpu().numpy()
            self._obs = np.copy(next_obs)
            return next_obs, reward, False, {}

    # generate a virtual system from the current, i.e., the task inference result
    env = LearnedEnv(agent.z)

    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []

    o = env.reset()
    next_o = None
    path_length = 0
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)

        next_o, r, d, env_info = env.step(a)
        # update the agent's current context
        if accum_context:
            agent.update_context([o, a, r, next_o, d, env_info])
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        path_length += 1
        o = next_o

        env_infos.append(env_info)
        if d:
            break

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )





def split_paths(paths):
    """
    Stack multiples obs/actions/etc. from different paths
    :param paths: List of paths, where one path is something returned from
    the rollout functino above.
    :return: Tuple. Every element will have shape batch_size X DIM, including
    the rewards and terminal flags.
    """
    rewards = [path["rewards"].reshape(-1, 1) for path in paths]
    terminals = [path["terminals"].reshape(-1, 1) for path in paths]
    actions = [path["actions"] for path in paths]
    obs = [path["observations"] for path in paths]
    next_obs = [path["next_observations"] for path in paths]
    rewards = np.vstack(rewards)
    terminals = np.vstack(terminals)
    obs = np.vstack(obs)
    actions = np.vstack(actions)
    next_obs = np.vstack(next_obs)
    assert len(rewards.shape) == 2
    assert len(terminals.shape) == 2
    assert len(obs.shape) == 2
    assert len(actions.shape) == 2
    assert len(next_obs.shape) == 2
    return rewards, terminals, obs, actions, next_obs


def split_paths_to_dict(paths):
    rewards, terminals, obs, actions, next_obs = split_paths(paths)
    return dict(
        rewards=rewards,
        terminals=terminals,
        observations=obs,
        actions=actions,
        next_observations=next_obs,
    )


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]
