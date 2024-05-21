import numpy as np
from typing import Any, Dict
from os import path
from gymnasium import spaces
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mujoco
from dm_control import suite
from dm_control.suite.walker import Physics, PlanarWalker
from dm_control.suite import common
from dm_control.utils import io as resources
from dm_control.utils import containers, rewards
from dm_control.rl import control

## ...


from . import register_env

_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = .025

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 1.0

# Horizontal speeds (meters/second) above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 8

_FLOOR_THICKNESS = .2  # a half of floor height


def get_reward(physics):
    """Returns a reward to the agent."""

    # rotation angle of z-axis of the floor around the y-axis
    # This is computed by applying the transformation quat -> euler.
    # TODO: take this as an argument instead of calculating it every time
    floor_quat = physics.named.model.geom_quat['floor']
    floor_x = physics.named.model.geom_pos['floor', 'x']
    torso_x = physics.named.data.xpos['torso', 'x']
    theta = 2. * np.arcsin(floor_quat[2])
    torso_relative_height = physics.torso_height() - (torso_x - floor_x) * np.tan(-theta)
    # correction of the relative height considering the 'thickness' of the floor
    # torso_relative_height = torso_relative_height + _FLOOR_THICKNESS * (1.0 - 1.0 / np.cos(theta))

    standing = rewards.tolerance(torso_relative_height,
                                 bounds=(_STAND_HEIGHT, float('inf')),
                                 margin=_STAND_HEIGHT / 2)

    # upright reward
    # 1  if torso's z-axis coincides with the z-axis of the world (global coordinate frame)
    # 1/2 if they are orthogonal
    # 0 if they are aligned in the opposite direction
    upright = (1 + physics.torso_upright()) / 2
    stand_reward = (3 * standing + upright) / 4

    # velocity along the floor, not along the horizontal direction
    velocity = physics.horizontal_velocity() / np.cos(theta)
    move_reward = rewards.tolerance(velocity,
                                    bounds=(_WALK_SPEED, float('inf')),
                                    margin=_WALK_SPEED / 2,
                                    value_at_margin=0.5,
                                    sigmoid='linear')

    return stand_reward * (5 * move_reward + 1) / 6


def read_model(model_filename):
    """Reads a model XML file and returns its contents as a string."""
    return resources.GetResource(model_filename)


# SUITE = containers.TaggedTasks()


def get_model_and_assets(task_index):
    """Returns a tuple containing the model XML string and a dict of assets."""
    return read_model(path.join(path.dirname(__file__), 'walker_assets/walker{}.xml'.format(task_index))), common.ASSETS


def walk(task_index, time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Walk task."""
    physics = Physics.from_xml_string(*get_model_and_assets(task_index))
    task = PlanarWalker(move_speed=_WALK_SPEED, random=np.random.RandomState(2024))
    # task.visualize_reward = True
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


@register_env('walker')
class WalkerSlopeWalkEnv:

    def __init__(self, randomize_tasks=False, n_tasks=125):
        random_state = np.random.RandomState(2023)
        self._env = suite.load('walker', 'walk', task_kwargs={
            'random': random_state})

        self._n_tasks = n_tasks
        # m = self._env.physics.named.model
        # self._param_types = ['body_mass', 'body_inertia', 'dof_damping', 'geom_friction', 'body_pos']
        # self._extended_param_types = self._param_types + ['geom_size']
        # self._init_params = {param_type: getattr(m, param_type) for param_type in self._param_types}
        # self._log_scale_lim = 3.0

        # self._multiplier_base = 1.3

        # self._tasks = self.sample_tasks_new(n_tasks=n_tasks, seed=2024)
        # print(self._tasks[0])
        obs_spec = self._env.observation_spec()
        act_spec = self._env.action_spec()
        obs_dim = sum([1 if not v.shape else v.shape[0]
                       for v in obs_spec.values()])
        # print(act_spec.minimum, act_spec.maximum)
        self.action_space = spaces.Box(low=act_spec.minimum,
                                       high=act_spec.maximum,
                                       shape=act_spec.shape,
                                       dtype=np.float32
                                       )
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(obs_dim,),
                                            dtype=np.float32
                                            )

        self._frames = []

        self._goal = None

        return

    def reset_task(self, idx):
        self._frames = []
        self._env = walk(task_index=idx)
        # params = self._tasks[idx]
        # self._param_vector = self._param_vectors[idx]
        # self.set_model_parameters_new(params)
        body_names = ['torso', 'right_thigh', 'left_thigh', 'right_leg', 'left_leg', 'right_foot', 'left_foot']
        m = self._env.physics.named.model
        for body_name in body_names:
            m.geom_rgba[body_name, :3] = np.random.rand(3)
            m.geom_rgba[body_name, 3] = .5

        '''
        m = self._env.physics.named.model
        for param_type in self._param_types:
            d = getattr(m, param_type)
            # print(d, d.shape)
            print(d)
        '''
        return

    def reset_model(self):
        time_step = self._env.reset()
        return step2state(time_step)

    def reset(self):
        return self.reset_model()

    def get_all_task_idx(self):
        return range(self._n_tasks)

    def step(self, u):
        # print([x.shape for x in self._env.physics._physics_state_items()])
        time_step = self._env.step(u)
        reward = get_reward(self._env.physics)
        done = False
        return step2state(time_step), reward, done, {}

    def close(self):
        return

    @property
    def dt(self):
        return self._env.control_timestep()

    def render(self):
        pixels = self._env.physics.render(camera_id=0, height=400, width=400)
        # print(pixels.shape)
        # img = Image.fromarray(pixels.astype('uint8'), 'RGB')
        # handle = ImageDraw.Draw(img)
        # font = ImageFont.truetype("arial.ttf", 20)
        # font = ImageFont.truetype("sans-serif.ttf", 20)
        # Add Text to an image
        # handle.text((200, 50), display_text, fill=(255, 255, 255), font=font)
        self._frames.append(pixels)
        return pixels

    '''
    def save_as_video(self, path):
        if len(self._frames) > 0:
            generate_video(frames=self._frames, framerate=int(1./self.dt), path=path)
    '''

    '''
    # TODO: use PyMJCF:
    def set_model_parameters(self, params):
        # configuration of model parameters
        # 1. robot parameters
        m = self._env.physics.named.model

        for param_type, param_dict in params.items():

            d = getattr(m, param_type)
            for param_name, param_val in param_dict.items():
                d[param_name] = param_val
        mujoco.mj_setConst(m)
        return

    def set_model_parameters_new(self, params):
        m = self._env.physics.named.model
        for param_type in self._param_types:
            p = getattr(m, param_type)
            # print('new param:', params[param_type])
            p = params[param_type]
            # setattr(m, param_type, params[param_type])
        p = getattr(m, 'geom_size')
        for body in ['left_foot', 'right_foot']:
            p[body] = params['geom_size'][body]
        return

    def sample_tasks_new(self, n_tasks, seed):
        np.random.seed(seed)
        # sample multiple tasks
        tasks = []

        log_scale_lim = self._log_scale_lim
        # See https://github.com/dennisl88/rand_param_envs/blob/master/rand_param_envs/base.py.
        for _ in range(n_tasks):
            task = {}
            for param_type in self._param_types:
                init_params = self._init_params[param_type]
                # uniform distribution over [-log scale lim, log scale lim)
                multiplier_exponents = np.random.uniform(-log_scale_lim, log_scale_lim, size=init_params.shape)
                task[param_type] = (self._multiplier_base ** multiplier_exponents) * init_params
                if param_type == 'body_inertia':
                    task[param_type][1:] = 100.

            left_foot_size = np.array([.05, np.random.uniform(0.25, 0.5), .0])
            right_foot_size = np.array([.05, np.random.uniform(0.25, 0.5), .0])
            task['geom_size'] = {'left_foot': left_foot_size, 'right_foot': right_foot_size}
            tasks.append(task)

        return tasks

    def sample_tasks(self, n_tasks, seed):
        np.random.seed(seed)
        # sample multiple tasks
        tasks = []
        # param_vectors = []
        for _ in range(n_tasks):
            # randomly generated parameters
            torso_mass = 4. * np.random.rand() + 8.
            thigh_mass = 1.8 * np.random.rand() + 3.6
            leg_mass = 1.2 * np.random.rand() + 2.4
            foot_mass = .6 * np.random.rand() + 1.8
            foot_length = .25 * np.random.rand() + .25
            # to be passed to the model
            task = {
                'body_mass': {
                    'torso': torso_mass,
                    'left_thigh': thigh_mass,
                    'right_thigh': thigh_mass,
                    'left_leg': leg_mass,
                    'right_leg': leg_mass,
                    'left_foot': foot_mass,
                    'right_foot': foot_mass
                },
                'geom_size': {
                    'left_foot': np.array([.05, foot_length, .0]),
                    'right_foot': np.array([.05, foot_length, .0])
                }
            }
            tasks.append(task)
            # param_vectors.append(np.array([torso_mass, thigh_mass, leg_mass, foot_mass, foot_length]))
        return tasks
    '''

    '''
    @property
    def parameters(self):
        return np.copy(self._param_vector)
    '''

    @property
    def control_range(self):
        return self.action_space.low, self.action_space.high

    @property
    def state_dimension(self):
        return self.observation_space.shape[0]

    @property
    def control_dimension(self):
        return self.action_space.shape[0]


def step2state(time_step):
    """
    Generation of state vectors. Each state vector consists of
    - (14-dim) orientations (2 x 7 links)
    - (1-dim)  height of the torso
    - (9-dim)  velocities
    so it is 24-dimensional.
    """
    return np.concatenate([np.atleast_1d(x) for x in time_step.observation.values()])


def generate_video(frames, framerate, path):
    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.

    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()

    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])

    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000. / framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    anim.save(path, writer='ffmpeg', dpi=dpi)

    return
