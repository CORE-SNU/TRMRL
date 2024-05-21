from abc import ABC, abstractmethod
import numpy as np
from numpy.linalg import matrix_rank
import torch
from torch import nn as nn

from torch.nn import functional as F
from torch.nn import SmoothL1Loss
from torch.optim import Adam
from control import dare, ctrb
from rlkit.torch.networks import Mlp, identity
from rlkit.torch import pytorch_util as ptu

from torch import nn as nn

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule



class DeterministicDynamics:
    """
    Implementation of a deterministic system model defined as

        x_{k+1} = x_k + dt f(x_k, u_k|p), k = 0, 1, ...

    where dt denotes the time step, f represents the vector field.
    """

    def __init__(self, state_dim, control_dim, parameter_dim, dt):
        self.state_dim = state_dim
        self.control_dim = control_dim

        # self.nominal_model = nominal_model
        self.vector_field: VectorField = VectorField(state_dim, control_dim, parameter_dim)
        # self.optimizer = Adam(self.vector_field.parameters(), lr=1e-3)

        self._dt = dt

    def __call__(self, states: torch.tensor, controls: torch.tensor, parameters: torch.tensor) -> torch.tensor:
        # TODO: remove these after debugging
        assert states.ndim == controls.ndim == parameters.ndim
        assert states.shape[:-1] == controls.shape[:-1] == parameters.shape[:-1]         # last dimension: features

        v = self.vector_field(states, controls, parameters)

        return states + self._dt * v

    @property
    def dt(self):
        return self._dt

    def __getattr__(self, attr):
        return getattr(self.vector_field, attr)

    '''
    def parameters(self):
        return self.vector_field.parameters()

    def update_model(self, states: torch.tensor, actions: torch.tensor, next_states: torch.tensor):
        # TODO: remove these after debugging
        assert states.ndim == actions.ndim == next_states.ndim
        assert states.shape[:-1] == actions.shape[:-1] == next_states.shape[:-1]  # last dimension: features
        input = torch.cat([states, actions], dim=-1)
        out = self.model_difference(input)
        # model uncertainty from data
        target = (next_states - states) / self.dt - self.nominal_model(states, actions)

        # simple least-square estimation loss for training the neural network
        # TODO: Does Huber loss have extra benefit?
        # loss = torch.mean(torch.sum((pred - truth) ** 2))
        loss_ftn = SmoothL1Loss()
        loss = loss_ftn(out, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    '''


class LinearDeterministicDynamics(DeterministicDynamics):
    """
    Implementation of a deterministic linear system model defined as

        x_{k+1} = x_k + dt (A(p) x_k + B(p) u_k), k = 0, 1, ...

    where dt denotes the time step.
    """
    def __init__(self, state_dim, control_dim, parameter_dim, dt, hidden_sizes):
        # TODO: network shape
        super().__init__(state_dim, control_dim, parameter_dim, dt)
        self.vector_field = LinearVectorField(state_dim, control_dim, parameter_dim, hidden_sizes=hidden_sizes)

    def system_parameters(self, parameters):
        # return [A(p) B(p)]
        return self.vector_field.system_parameters(parameters)

    def __getattr__(self, attr):
        return getattr(self.vector_field, attr)

    def ctrb_flags(self, parameters):
        """
        (damped) controllability check
        """
        sys_params = self.system_parameters(parameters=parameters).cpu().numpy()
        As, Bs = sys_params[..., :self.state_dim], sys_params[..., self.state_dim:]
        batch_size = parameters.shape[0]
        controllability = []
        for i in range(batch_size):
            A, B = As[i], Bs[i]
            ctrb_mat_rank = matrix_rank(ctrb(A, B))
            is_controllable = (ctrb_mat_rank == self.state_dim)
            controllability.append(is_controllable)
        return controllability


    def optimal_solutions(self, parameters, state_cost_weights, control_cost_weights, discount):
        """
        Computation of the current model's true value functions & optimal policies.
        This is done by solving discrete algebraic Riccati equation.
        See https://python-control.readthedocs.io/en/0.9.4/generated/control.dare.html.

        This is used to check if the certainty equivalent controller & value function match the true ones.
        """

        Q = state_cost_weights
        R = control_cost_weights

        gamma_sqrt = discount ** .5

        sys_params = self.system_parameters(parameters=parameters).cpu().numpy()
        As, Bs = sys_params[..., :self.state_dim], sys_params[..., self.state_dim:]
        # Only the first dimension of the parameters is assumed to indicate the batch size.
        # TODO: generalization?
        batch_size = parameters.shape[0]

        Xs = []
        Gs = []

        for i in range(batch_size):
            A, B = As[i], Bs[i]

            A = gamma_sqrt * (np.eye(self.state_dim) + self._dt * A)
            B = gamma_sqrt * self._dt * B
            # X: value function / L: eigenvalues of the closed-loop system A - B * G / G: control gain matrix
            X, L, G = dare(A=A, B=B, Q=Q, R=R)
            Xs.append(X)
            Gs.append(G)

        return Xs, Gs


class MLPDeterministicDynamics(DeterministicDynamics):
    def __init__(self, state_dim, control_dim, parameter_dim, dt, hidden_sizes, hidden_activation=F.relu, output_activation=identity):
        # TODO: GeLU?
        super().__init__(state_dim, control_dim, parameter_dim, dt)
        self.vector_field = MLPVectorField(state_dim, control_dim, parameter_dim, hidden_sizes, hidden_activation, output_activation)


class VectorField(nn.Module):
    """
    Implementation of the vector field model f(x, u|p) which defines the system dynamics

        x_{k+1} = x_k + dt f(x_k, u_k|p), k = 0, 1, ...

    where dt: time step & p: model parameter
    """
    def __init__(self, state_dim, control_dim, parameter_dim):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.param_dim = parameter_dim

    def forward(self, states, controls, parameters):
        raise NotImplementedError


class LinearVectorField(VectorField):
    """
    Implementaion of the linear vector field 
    
        f(x, u|p) = A(p, theta)x + B(p, theta)u

    where A(p, theta) & B(p, theta): system parameters given as the functions of the model parameters p & learnable parameters theta.
    """
    def __init__(self, state_dim, 
                 control_dim, 
                 parameter_dim, 
                 hidden_sizes,
                 init_w=3e-3,
                 hidden_init=ptu.fanin_init,
                 b_init_value=0.1,                  
                 hidden_activation=F.relu, 
                 output_activation=identity
                 ):
        super().__init__(state_dim, control_dim, parameter_dim)

        sys_param_dim = state_dim * (state_dim + control_dim)

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.fc_layers = nn.ModuleList([])

        in_size = parameter_dim
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fc_layers.append(fc)

        # self.fc_layers.append(nn.Linear(parameter_dim, 128))
        # self.fc_layers.append(nn.Linear(128, 128))
        # self.fc_layers.append(nn.Linear(128, 128))

        self.last_fc = nn.Linear(in_size, sys_param_dim)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)


    def forward(self, states, actions, parameters):
        # z = [x u]^T
        z = torch.cat((states, actions), dim=-1)

        # compute [A(p, theta) B(p, theta)]
        h = parameters
        for layer in self.fc_layers:
            h = F.relu(layer(h))

        batch_dim = h.shape[:-1]
        z_dim = self.state_dim + self.control_dim
        sys_params = self.output_activation(self.last_fc(h).view(*batch_dim, self.state_dim, z_dim))

        # vector field = [A(p, theta) B(p, theta)] @ z

        # (batch size, state dim, state + action dim) @ (batch_size, state + action dim, 1)
        # -> (batch size, state dim, 1) -> (batch size, state dim)
        output = torch.squeeze(sys_params @ torch.unsqueeze(z, -1), -1)

        return output
    
    def system_parameters(self, parameters):
        # compute [A(p, theta) B(p, theta)]
        h = parameters
        for layer in self.fc_layers:
            h = self.hidden_activation(layer(h))

        batch_dim = h.shape[:-1]
        z_dim = self.state_dim + self.control_dim
        sys_params = self.output_activation(self.last_fc(h).view(*batch_dim, self.state_dim, z_dim))

        return sys_params.detach()
    

class MLPVectorField(Mlp):
    def __init__(self,
                 state_dim, 
                 control_dim, 
                 parameter_dim, 
                 hidden_sizes, 
                 hidden_activation=F.relu, 
                 output_activation=identity
                 ):
        self.save_init_params(locals())

        Mlp.__init__(self,
                     input_size=state_dim+control_dim+parameter_dim,
                     output_size=state_dim,
                     hidden_sizes=hidden_sizes,
                     hidden_activation=hidden_activation,
                     output_activation=output_activation
                     )


    def forward(self, states, actions, parameters):
        input = torch.cat([states, actions, parameters], dim=-1)
        return super().forward(input)
    

class RewardFunction(nn.Module):
    def __init__(self, state_dim, control_dim, parameter_dim):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.param_dim = parameter_dim

    def forward(self, states, controls, parameters):
        raise NotImplementedError


class MLPReward(Mlp):
    """
    Separate model for tasks' reward functions. The learnable parameters of the model are not shared with the dynamics model.
    """
    def __init__(self,
                 state_dim, 
                 control_dim, 
                 parameter_dim, 
                 hidden_sizes, 
                 hidden_activation=F.relu, 
                 output_activation=identity
                 ):
        self.save_init_params(locals())

        Mlp.__init__(self,
                     input_size=state_dim+control_dim+parameter_dim,
                     output_size=1,
                     hidden_sizes=hidden_sizes,
                     hidden_activation=hidden_activation,
                     output_activation=output_activation
                     )

        
    def forward(self, states, actions, parameters):
        input = torch.cat([states, actions, parameters], dim=-1)
        return super().forward(input)
    

class MLPDeterministicModel(Mlp):
    """
    Implementation of the dynamics & reward function model approximated by a (dual-head) multilayer perceptron.
    The dynamics is given as

        x_{k+1} = x_k + dt f(x_k, u_k|p), k = 0, 1, ...

    where dt denotes the time step, f represents the vector field (parameterized by the non-learnable parameter p).
    """
    def __init__(self,
                 state_dim, 
                 control_dim, 
                 parameter_dim, 
                 dt,
                 hidden_sizes, 
                 hidden_activation=F.relu, 
                 output_activation=identity,
                 hidden_init=ptu.fanin_init,
                 b_init_value=0.1,
                 init_w_dynamics=3e-3,              # for dynamics head
                 init_w_reward=3e-3                 # for reward function head
                 ):
        self.save_init_params(locals())

        # TODO: how to incorporate the knowledge about dynamics / reward function into the model?
        Mlp.__init__(self,
                     input_size=state_dim+control_dim+parameter_dim,
                     output_size=hidden_sizes[-1],      # hidden_sizes: for the layers before branching
                     hidden_sizes=hidden_sizes[:-1],
                     hidden_activation=hidden_activation,
                     output_activation=output_activation,
                     hidden_init=ptu.fanin_init,
                     b_init_value=b_init_value
                     )
        # final layer before branching

        self._dt = dt

        hidden_init(self.last_fc.weight)
        self.last_fc.bias.data.fill_(b_init_value)

        # branch 1: for dynamics
        self.fc_dynamics = nn.Linear(hidden_sizes[-1], state_dim)
        self.fc_dynamics.weight.data.uniform_(-init_w_dynamics, init_w_dynamics)
        self.fc_dynamics.bias.data.uniform_(-init_w_dynamics, init_w_dynamics)

        # branch 2: for reward function
        self.fc_reward_function = nn.Linear(hidden_sizes[-1], 1)
        self.fc_reward_function.weight.data.uniform_(-init_w_reward, init_w_reward)
        self.fc_reward_function.bias.data.uniform_(-init_w_reward, init_w_reward)

    def forward(self, states, actions, parameters):
        input = torch.cat([states, actions, parameters], dim=-1)
        h = super().forward(input)
        h = self.hidden_activation(h)
        out_dynamics = self.output_activation(self.fc_dynamics(h))
        out_reward = self.output_activation(self.fc_reward_function(h))
        return states + self._dt * out_dynamics, out_reward