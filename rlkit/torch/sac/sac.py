from collections import OrderedDict
import time
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm
from torch.distributions import multivariate_normal
from rlkit.torch.sac.mcmc import markov_chain_monte_carlo
from rlkit.torch.dynamics import LinearDeterministicDynamics


class PEARLSoftActorCritic(MetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            model_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,
            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            init_w=None,        # not used within the algorithm; already used when defining the networks
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )

        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.recurrent = recurrent
        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context

        self.qf1, self.qf2, self.vf, self.model = nets[1:]
        self.target_vf = self.vf.copy()

        self.policy_optimizer = optimizer_class(
            self.agent.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.agent.context_encoder.parameters(),
            lr=context_lr,
        )

        self.model_optimizer = optimizer_class(
            self.model.parameters(),
            lr=model_lr,

        )

        '''
        self.latent_optimizer = optimizer_class(
            self.latent_encoder.parameters(),
            lr=context_lr,
        )
        '''
    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.vf, self.target_vf, self.model]

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        return context

    ##### Training #####

    def _do_pretraining(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            decoder_loss = self._take_prestep(indices, context)

            # stop backprop
            self.agent.detach_z()
        return decoder_loss

    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)
        
        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))
        # if iter_pre % 150 == 0:
        #     self._take_prestep(indices, context_batch)
        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            # self._take_prestep(indices, context_batch)
            self._take_step(indices, context)

            # stop backprop
            self.agent.detach_z()

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _take_step(self, indices, context):
        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terminals = self.sample_sac(indices)

        # TODO: linear policy for LQR
        # run inference in networks
        policy_outputs, task_z = self.agent(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        rewards_flat = rewards.view(t * b, -1)

        # +------------------------------------------ decoder loss ------------------------------------------+
        # (task, latent_dim) -> (task, batch, latent_dim) -> (task * batch, latent_dim)
        # We use fixed set of z's here!
        # TODO: same loss for enc & dec?
        # self.decoder_optimizer.zero_grad()
        # self.latent_optimizer.zero_grad()
        # # TODO: do not fix latent vectors for training tasks
        # z_train = self.z_train[indices].unsqueeze(1).repeat(1, b, 1).view(t * b, -1)
        # z_train = self.latent_encoder(z_train)

        # dec_pred = self.decoder(obs, actions, task_z.detach())
        # dec_loss = torch.mean((dec_pred - next_obs) ** 2)        # simple L^2 loss
        # dec_kl_loss = torch.mean((1 + dec_pred ** 2)/2-0.5)
        # _dec_loss = dec_loss + dec_kl_loss
        # _dec_loss.backward()

        '''
        # Leaning the dynamics model via supervised learning
        # By doing so, the model is updated twice within a single step.
        # This is needed as the initial value function is close to 0, which may slow down the model learning.
        self.dynamics_optimizer.zero_grad()
        next_obs_pred = self.dynamics(obs, actions, task_z.detach())
        # Since the vector field of the dynamics is being approximated, we divide the loss by dt.
        dynamics_loss = torch.mean((next_obs_pred - next_obs)) / self.dynamics.dt
        dynamics_loss.backward()
        self.dynamics_optimizer.step()
        # self.latent_optimizer.step()
        # self.decoder_optimizer.step()
        '''

        # +--------------------------------------------------------------------------------------------------+
        # TODO: use the reward function structure
        # Q and V networks
        # encoder will only get gradients from Q nets

        # Since training the policy & task can be done across all latent vectors, we randomly sample z's.

        n_policy_learning_steps = 5


        ub = ptu.from_numpy(self.env.action_space.high).detach()
        lb = ptu.from_numpy(self.env.action_space.low).detach()
        
        z_rand = ptu.normal(mean=0., std=1, size=(t * b, self.latent_dim)).detach()
        # with torch.no_grad():
        #     z_rand = self.latent_encoder(z_rand)
        # z_rand += z_train
        

        for _ in range(n_policy_learning_steps):
            # data perturbation
            obs_perturbed = obs + ptu.normal(mean=0., std=0.1, size=obs.shape).detach()
            actions_perturbed = torch.clamp(actions + ptu.normal(mean=0., std=0.1, size=actions.shape).detach(), min=lb, max=ub)

            q1_pred = self.qf1(obs_perturbed, actions_perturbed, z_rand)
            q2_pred = self.qf2(obs_perturbed, actions_perturbed, z_rand)
            v_pred = self.vf(obs_perturbed, z_rand)
            # get targets for use in V and Q updates

            # with torch.no_grad():
            #     target_v_values_true = self.target_vf(next_obs, task_z)
            #     target_v_values = target_v_values_true.detach()

            # model-based loss?
            # This makes sense when all tasks share the same single reward function.
            with torch.no_grad():
                # next_obs_pred = self.decoder(obs, actions, z_rand)
                next_obs_pred, reward_pred = self.model(obs_perturbed, actions_perturbed, z_rand)
                target_v_values_true = self.target_vf(next_obs_pred, z_rand)
                # target_v_values = target_v_values_true.detach()

            # KL constraint on z if probabilistic
            # self.context_optimizer.zero_grad()
            # if self.use_information_bottleneck:
            #     kl_div = self.agent.compute_kl_div()
            #     kl_loss = self.kl_lambda * kl_div
            #     kl_loss.backward(retain_graph=True)

            self.qf1_optimizer.zero_grad()
            self.qf2_optimizer.zero_grad()
            
            # rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
            # scale rewards for Bellman update
            reward_pred = reward_pred * self.reward_scale
            terms_flat = terminals.view(self.batch_size * num_tasks, -1)        # terminal
            q_target = reward_pred + (1. - terms_flat) * self.discount * target_v_values_true
            qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
            qf_loss.backward()
            # TODO: optimize at once?
            self.qf1_optimizer.step()
            self.qf2_optimizer.step()

            # compute min Q on the new actions

            # z_rand instead of task_z
            in_ = torch.cat([obs_perturbed, z_rand], dim=1)
            policy_outputs_rand = self.agent.policy(in_, reparameterize=True, return_log_prob=True)
            new_actions_rand, policy_mean_rand, policy_log_std_rand, log_pi_rand = policy_outputs_rand[:4]

            min_q_new_actions = self._min_q(obs_perturbed, new_actions_rand, z_rand)

            # vf update
            v_target = min_q_new_actions - log_pi_rand
            vf_loss = self.vf_criterion(v_pred, v_target.detach())

            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()
            self._update_target_network()

            # policy update
            # n.b. policy update includes dQ/da
            log_policy_target = min_q_new_actions
            policy_loss = (log_pi_rand - log_policy_target).mean()

            mean_reg_loss = self.policy_mean_reg_weight * (policy_mean_rand ** 2).mean()
            std_reg_loss = self.policy_std_reg_weight * (policy_log_std_rand ** 2).mean()
            pre_tanh_value = policy_outputs_rand[-1]
            pre_activation_reg_loss = self.policy_pre_activation_weight * (
                (pre_tanh_value ** 2).sum(dim=1).mean()
            )
            policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
            policy_loss = policy_loss + policy_reg_loss

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
        # self._update_target_network()

        # +------------------------------------------ encoder loss ------------------------------------------+
        # encoder update
        # We do not differentiate the target value function w.r.t. its 2nd argument.
        # Here the target value function just serves as a proxy for the optimal value function.

        if self.use_information_bottleneck:
            # TODO: I think this is needed...
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div

        # Again, task_z is latent variable inferenced by encoder.
        # target_v_values : V(s_pred(s,a ; z_rand), z_rand) :
        # target_v_values_pred : V(s_pred(s,a ; z_infer), z_infer(no_grad)) : from another task
        task_z_no_grad = task_z.detach()
        # target_v_values_pred = self.target_vf(self.decoder(obs, actions, task_z), task_z_no_grad)
        next_obs_pred_enc, reward_pred_enc = self.model(obs, actions, task_z)
        target_v_values_pred = self.target_vf(next_obs_pred_enc, task_z_no_grad)
        target_v_values = self.target_vf(next_obs, task_z_no_grad)
        # TODO: automatic adjustment of multiplier
        # TODO: add value iteration error?
        rewards_flat = rewards_flat * self.reward_scale
        reward_pred_enc = reward_pred_enc * self.reward_scale
        enc_loss = torch.mean((rewards_flat + self.discount * target_v_values - (reward_pred_enc + self.discount * target_v_values_pred)) ** 2) + kl_loss        # L^2 loss
        self.context_optimizer.zero_grad()
        # self.decoder_optimizer.zero_grad()
        self.model_optimizer.zero_grad()
        enc_loss.backward()
        self.context_optimizer.step()
        self.model_optimizer.step()

        '''
        is_lqr = isinstance(self.dynamics, LinearDeterministicDynamics) and hasattr(self.env, 'system_parameters')
        if is_lqr:
            # only LQR case
            # TODO: only consider the class of linear policies (& quadratic value functions)?
            # (# tasks,) + matrix shape
            task_z_no_duplicate = task_z.view(t, b, -1)
            task_z_no_duplicate = task_z_no_duplicate[:, 0, :]

            
            model_difference = self.env.system_parameters(indices) - self.dynamics.system_parameters(task_z_no_duplicate).detach().cpu().numpy()
            # average model error measured as 2-norms
            model_error = np.mean([np.linalg.norm(model_difference[i], ord=2) for i in range(t)])

            ctrb_flags = self.dynamics.ctrb_flags(task_z_no_duplicate)
            if all(ctrb_flags):
                Xs, Gs = self.env.optimal_solutions(indices, self.discount)
                Xs_model, Gs_model = self.dynamics.optimal_solutions(task_z_no_duplicate, self.env.state_cost_weights, self.env.control_cost_weights, self.discount)

                value_function_error = np.mean([np.linalg.norm(Xs[i] - Xs_model[i], ord=2) for i in range(t)])
                Xs_evaluated = self.env.evaluate_gains(indices, Gs_model, self.discount)
                policy_error = np.mean([np.linalg.norm(Xs[i] - Xs_evaluated[i], ord=2) for i in range(t)])
            else:
                print('The learned model is not controllable...evaluation step skipped')
                value_function_error = np.nan
                policy_error = np.nan
        '''
        # self.decoder_optimizer.step()
        # +--------------------------------------------------------------------------------------------------+

        # for name, p in self.qf1.named_parameters():
        #     p.requires_grad = True
        # for name, p in self.qf2.named_parameters():
        #     p.requires_grad = True

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig

                '''
                # only for LQR
                if is_lqr:
                    self.eval_statistics['model error (model)'] = model_error
                    self.eval_statistics['model error (value function)'] = value_function_error
                    self.eval_statistics['model error (policy)'] = policy_error
                '''
                # self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                # self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['Encoder Loss'] = np.mean(ptu.get_numpy(enc_loss))
            # self.eval_statistics['Decoder Loss'] = np.mean(ptu.get_numpy(dec_loss))
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
            model=self.model.state_dict()
        )
        return snapshot

