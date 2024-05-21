import os, shutil
import os.path as osp
import pickle
import json
import numpy as np
import click
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from rlkit.envs import ENVS
# from rlkit.envs.wrappers import NormalizedBoxEnv, CameraWrapper
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.agent import PEARLAgent
from configs.default import default_config
from caml import deep_update_dict
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.util import rollout


def sim_policy(variant, path_to_exp, num_trajs=1, deterministic=False, save_video=False):
    '''
    simulate a trained policy adapting to a new task
    optionally save videos of the trajectories - requires ffmpeg

    :variant: experiment configuration dict
    :path_to_exp: path to exp folder
    :num_trajs: number of trajectories to simulate per task (default 1)
    :deterministic: if the policy is deterministic (default stochastic)
    :save_video: whether to generate and save a video (default False)
    '''

    # create multi-task environment and sample tasks
    # env = CameraWrapper(NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params'])), variant['util_params']['gpu_id'])
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    eval_tasks=list(tasks[-variant['n_eval_tasks']:])
    print(obs_dim, action_dim)
    print('testing on {} test tasks, {} trajectories each'.format(len(eval_tasks), num_trajs))

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    reward_dim = 1
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    encoder_input_size = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=encoder_input_size,
        output_size=context_encoder,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )
    # deterministic eval
    if deterministic:
        agent = MakeDeterministic(agent)

    # load trained weights (otherwise simulate random policy)
    context_encoder.load_state_dict(torch.load(os.path.join(path_to_exp, 'context_encoder.pth')))
    policy.load_state_dict(torch.load(os.path.join(path_to_exp, 'policy.pth')))

    # loop through tasks collecting rollouts
    all_rets = []
    # video_frames = []
    sample_frames = []
    for idx in eval_tasks:
        video_frames = []
        env.reset_task(idx)
        agent.clear_z()
        paths = []
        for n in range(num_trajs):
            if n >= variant['algo_params']['num_exp_traj_eval']:
                agent.infer_posterior(agent.context)
            path = rollout(env, agent, max_path_length=variant['algo_params']['max_path_length'], accum_context=True, save_frames=save_video)
            print('[task {} ep {}] return = {}'.format(idx, n, sum(path['rewards'])))
            paths.append(path)
            if save_video and (n == 0 or n == num_trajs - 1):
                video_frames += [t['frame'] for t in path['env_infos']]
            if n == 0:
                sample_frames.append(video_frames[15])
        all_rets.append([sum(p['rewards']) for p in paths])

        if save_video:
            video_filename=os.path.join(path_to_exp, 'video{}.mp4'.format(idx))
            generate_video(video_frames, framerate=int(1./env.dt), path=video_filename)
        ''' 
        # save frames to file temporarily
        temp_dir = os.path.join(path_to_exp, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        for i, frm in enumerate(video_frames):
            frm.save(os.path.join(temp_dir, '%06d.jpg' % i))

        video_filename=os.path.join(path_to_exp, 'video{}.mp4'.format(idx))
        # run ffmpeg to make the video
        os.system('ffmpeg -i {}/%06d.jpg -vcodec mpeg4 {}'.format(temp_dir, video_filename))
        # delete the frames
        shutil.rmtree(temp_dir)
        '''
    
    sample_frame = sample_frames[0]
    height, width, _ = sample_frame.shape
    # print(height, width)
    dpi = 70
    # orig_backend = matplotlib.get_backend()
    # matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    nrows, ncols = 1, len(eval_tasks)
    # fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * width / dpi, nrows * height / dpi), dpi=dpi, constrained_layout=True)
    fig = plt.figure(figsize=(ncols * (width / dpi), nrows * (height / dpi)), dpi=dpi)
    ax = [fig.add_subplot(1, ncols, i+1) for i in range(ncols)]
    # matplotlib.use(orig_backend)  # Switch back to the original backend.
    
    im_list = []
    for r in range(nrows):
        for c in range(ncols):
            ax[c].set_axis_off()
            ax[c].set_aspect('equal')
            # ax.set_position([0, 0, 1, 1])
            ax[c].imshow(sample_frames[c + r * ncols])
            # im_list.append()
    
    # fig.tight_layout()
    fig.subplots_adjust(left=0, right=0.99, top=1, bottom=0)
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig('{}_test_tasks.pdf'.format(variant['env_name']))

    # compute average returns across tasks
    n = min([len(a) for a in all_rets])
    rets = [a[:n] for a in all_rets]
    rets = np.mean(np.stack(rets), axis=0)
    for i, ret in enumerate(rets):
        print('trajectory {}, avg return: {} \n'.format(i, ret))


def generate_video(frames, framerate, path):
    """
    Forked from
    https://github.com/google-deepmind/dm_control/blob/main/tutorial.ipynb
    """
    plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
    height, width, _ = frames[0].shape
    dpi = 240
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    # matplotlib.use(orig_backend)  # Switch back to the original backend.
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
    anim.save(path, writer='ffmpeg')
    
    return 




@click.command()
@click.argument('config', default=None)
@click.argument('path', default=None)
@click.option('--num_trajs', default=3)
@click.option('--deterministic', is_flag=True, default=False)
@click.option('--video', is_flag=True, default=False)
def main(config, path, num_trajs, deterministic, video):
    # os.environ['MUJOCO_GL'] = 'egl'
    variant = default_config
    if config:
        with open(osp.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    sim_policy(variant, path, num_trajs, deterministic, video)


if __name__ == "__main__":
    main()
