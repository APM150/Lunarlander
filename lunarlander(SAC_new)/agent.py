import os
import torch as T
import torch.nn.functional as F
import numpy as np
from replayBuffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8],
                 env=None, gamma=0.99, n_actions=2, max_size=1000000, polyak=0.995,
                 batch_size=256, reward_scale=3, entropyReg=1):
        self.gamma = gamma
        self.polyak = polyak
        self.entropyReg = entropyReg
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions,
                                  name='actor', max_action=env.action_space.high)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                      name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                      name='critic_2')
        self.target_critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                             name='target_critic_1')
        self.target_critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                             name='target_critic_2')

        self.scale = reward_scale
        self.update_target_network_parameters(polyak=0)


    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_target_network_parameters(self, polyak=None):
        if polyak is None:
            polyak = self.polyak

        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        for name in target_critic_1_state_dict:
            target_critic_1_state_dict[name] = polyak*target_critic_1_state_dict[name].clone() + \
                                               (1-polyak)*critic_1_state_dict[name].clone()
            target_critic_2_state_dict[name] = polyak*target_critic_2_state_dict[name].clone() + \
                                               (1-polyak)*critic_2_state_dict[name].clone()

        self.target_critic_1.load_state_dict(target_critic_1_state_dict)
        self.target_critic_2.load_state_dict(target_critic_2_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done, dtype=T.float).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        # update critic (Q-function)
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        actions, log_probs = self.actor.sample_normal(state_, reparameterize=False)
        log_probs = log_probs.view(-1)
        with T.no_grad():
            q1_new_policy = self.target_critic_1.forward(state_, actions)
            q2_new_policy = self.target_critic_1.forward(state_, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        y = self.scale*reward + self.gamma*(1-done)*(critic_value - self.entropyReg*log_probs)
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = F.mse_loss(q1_old_policy, y)
        critic_2_loss = F.mse_loss(q2_old_policy, y)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # update actor (policy)
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = self.entropyReg*log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # update target network
        self.update_target_network_parameters()