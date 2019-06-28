import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import yaml
import string as String
from spacy.lang.en.stop_words import STOP_WORDS
from textworld import EnvInfos
from sklearn.preprocessing import OneHotEncoder
from agents.A2Cagent.network import FCNetwork, RNNNetwork
from os.path import realpath as path

import warnings
warnings.simplefilter('ignore')


class CustomAgent:
    def __init__(self):
        # agent configuration
        with open('/home/nik-96/Documents/git/TextWorld_Challenge/agents/A2Cagent/config.yaml') as config_file:
            self.config = yaml.safe_load(config_file)
        try:
            self.vector_size = int(self.config['training']['vector_size'])
            self.hidden_size = int(self.config['training']['hidden_size'])
            self.output_size = int(self.config['training']['output_size'])
            self.gamma = float(self.config['training']['gamma'])
            self.act_lr = float(self.config['training']['actor_learning_rate'])
            self.critic_lr = float(self.config['training']['critic_learning_rate'])
            self.batch_size = int(self.config['training']['batch_size'])
            self.max_nb_steps_per_episode = int(self.config['training']['max_nb_steps_per_episode'])
            self.entropy_coef = float(self.config['training']['entropy_coefficient'])
            self.device = str(self.config['training']['device'])
            self.use_pretrained_model = bool(self.config["testing"]["use_pretrained_model"])
            self.test_time = bool(self.config["testing"]["test_time"])
        except KeyError:
            print("Check and double check config.yaml file for typos and errors, parameters will be set to default")
            # set default parameters
            self.vector_size = 50
            self.hidden_size = 200
            self.output_size = 100
            self.gamma = 0.99
            self.act_lr = 1e-4
            self.critic_lr = 1e-4
            self.batch_size = None
            self.max_nb_steps_per_episode = 50
            self.entropy_coef = 0.1
            self.device = 'cpu'
            self.use_pretrained_model = False
            self.test_time = False

        self.params = {'vector_size': self.vector_size,
                       'hidden_size': self.hidden_size,
                       'output_size': self.output_size,
                       'gamma': self.gamma,
                       'actor_learning_rate':  self.act_lr,
                       'critic_learning_rate': self.critic_lr,
                       'batch_size': self.batch_size,
                       'max_nb_steps_per_episode': self.max_nb_steps_per_episode,
                       'entropy_coef': self.entropy_coef,
                       'self.device': self.device
        }

        with open("/home/nik-96/Documents/git/TextWorld_Challenge/vocab.txt", 'r') as file:
            vocab = file.read().split('\n')

        print('[AGENT INFO] Loading word vectors')
        # word vectors: 08.04 -> glove 50d | 07.05 -> fasttext wiki-news-300d-1M.vec
        # for my laptop the path is /media/nik/hdd-data/datasets/glove/
        # and for my work pc the path is /home/nik-96/Documents/datasets/glove/
        # with open("/media/nik/hdd-data/datasets/glove/glove.6B.{}d.txt".format(self.vector_size)) as file:
        # 19.05 see transform_word_vectors.py, where it's formed word_vectors file from vocab.txt file
        with open('/home/nik-96/Documents/git/TextWorld_Challenge/agents/A2Cagent/fasttext_word_vectors.vec') as file:
            #                     word            :      vector
            self.wordVectors = {word_vector.split(' ')[0]: np.array(word_vector.split(' ')[1:], dtype='float')
                                for word_vector in file.read().split('\n')}
            self.wordVectors.pop('')
        print('[AGENT INFO] Word vectors were loaded')
        # word: id
        self.vocab = {word: i for i, word in enumerate(vocab)}

        self.trans_table = str.maketrans('\n', ' ', String.punctuation)
        self.StringVectors = {}

        self.recipe_batch = None

        # a simple model, TODO last 4 observations in obs model
        # self.obs_model = FCNetwork(2*self.vector_size,
        #                            2*self.hidden_size,
        #                            self.output_size,
        #                            self.device).to(self.device)
        self.obs_hidden_state = (torch.autograd.Variable(torch.zeros(self.batch_size, self.hidden_size)),
                                 torch.autograd.Variable(torch.zeros(self.batch_size, self.hidden_size)))
        self.obs_model = RNNNetwork(2*self.vector_size,
                                    self.hidden_size,
                                    self.output_size).to(self.device)
        self.act_model = FCNetwork(self.vector_size,
                                   self.hidden_size,
                                   self.output_size).to(self.device)
        self.state_value = nn.Linear(self.output_size, 1)

        if self.use_pretrained_model and self.test_time:
            self.load_pretrained_model('/home/nik-96/Documents/git/TextWorld_Challenge/agents/A2Cagent' +
                                       '/models/0610_16:00_episode_100')

        self.act_opt = torch.optim.Adam(self.act_model.parameters(), lr=self.act_lr)
        self.critic_opt = torch.optim.Adam(list(self.state_value.parameters()) + list(self.obs_model.parameters()),
                                           lr=self.critic_lr)

    def select_additional_infos(self) -> EnvInfos:
        """
        Returns what additional information should be made available at each game step.

        Requested information will be included within the `infos` dictionary
        passed to `CustomAgent.act()`. To request specific information, create a
        :py:class:`textworld.EnvInfos <textworld.envs.wrappers.filter.EnvInfos>`
        and set the appropriate attributes to `True`. The possible choices are:

        * `description`: text description of the current room, i.e. output of the `look` command;
        * `inventory`: text listing of the player's inventory, i.e. output of the `inventory` command;
        * `max_score`: maximum reachable score of the game;
        * `objective`: objective of the game described in text;
        * `entities`: names of all entities in the game;
        * `verbs`: verbs understood by the the game;
        * `command_templates`: templates for commands understood by the the game;
        * `admissible_commands`: all commands relevant to the current state;

        In addition to the standard information, game specific information
        can be requested by appending corresponding strings to the `extras`
        attribute. For this competition, the possible extras are:

        * `'recipe'`: description of the cookbook;
        * `'walkthrough'`: one possible solution to the game (not guaranteed to be optimal);

        Example:
            Here is an example of how to request information and retrieve it.

            # >>> from textworld import EnvInfos
            # >>> request_infos = EnvInfos(description=True, inventory=True, extras=["recipe"])
            ...
            # >>> env = gym.make(env_id)
            # >>> ob, infos = env.reset()
            # >>> print(infos["description"])
            # >>> print(infos["inventory"])
            # >>> print(infos["extra.recipe"])

        Notes:
            The following information *won't* be available at test time:

            * 'walkthrough'
        """
        request_infos = EnvInfos()
        request_infos.has_won = True
        request_infos.has_lost = True
        request_infos.admissible_commands = True
        request_infos.extras = ["recipe"]
        return request_infos

    def load_pretrained_model(self, load_from: str):
        """
        Load pretrained checkpoint from file.
        :param load_from: path to directory (without '/' in the end), that contains obs model and act model
        """
        print(f"\n[INFO] Loading model from {load_from}\n")
        try:

            obs_state_dict = torch.load(load_from + '/obs_model.pt', map_location='cpu')
            act_state_dict = torch.load(load_from + '/act_model.pt', map_location='cpu')
            state_value_dict = torch.load(load_from + '/state_value.pt', map_location='cpu')
            self.obs_model.load_state_dict(obs_state_dict)
            self.act_model.load_state_dict(act_state_dict)
            self.state_value.load_state_dict(state_value_dict)
        except:
            print("[INFO] Failed to load checkpoint...")

    def save_model(self, save_path: str):
        """
        Save checkpoint to file.
        :param save_path: path to directory (without '/' in the end), where to save models
        """
        print(f"\n[INFO] Saving model to {path(save_path) + '/'}")
        try:
            torch.save(self.obs_model.state_dict(), save_path + '/obs_model.pt')
            torch.save(self.act_model.state_dict(), save_path + '/act_model.pt')
            torch.save(self.state_value.state_dict(), save_path + '/state_value.pt')
        except:
            print("[INFO] Failed to save checkpoint...")

    def prepare_string(self, string):
        """
        This function preprocesses string feedback from environment and return torch tensor of average token embeddings
        :param string: str feedback from environment
        :return: torch.tensor
        """
        string_vector_ = self.StringVectors.get(string, None)
        if string_vector_ is not None:
            return torch.FloatTensor(string_vector_)

        prep_string = re.split(' ', re.sub('[ ]+', ' ', string.lower().translate(self.trans_table)))
        string_vector = np.array([self.wordVectors[word] for word in prep_string
                                  if word in self.wordVectors and word not in STOP_WORDS])
        string_vector = np.mean(string_vector, axis=0)
        self.StringVectors[string] = string_vector

        return torch.FloatTensor(string_vector)

    def get_logits(self, states, admissible_commands, recipes):
        """
        Getting logits from two neural networks
        :param state: string - observation from environment
        :param admissible_commands: list of strings - possible commands in state
        :param recipe: string - recipe description
        :return: list of logits from networks, length of returned list equals to length of admissible_commands list
        """
        prep_obs = torch.stack([self.prepare_string(state) for state in states]).to(self.device)
        print("prep_obs shape", prep_obs.shape)
        self.recipe_vectors = torch.stack([self.prepare_string(recipe).to(self.device) for recipe in recipes])
        print("recipe vectors shape", self.recipe_vectors.shape)
        print("concat shape", torch.cat([prep_obs, self.recipe_vectors]).shape)
        observation = torch.cat([prep_obs, self.recipe_vectors], dim=1)
        print("observation shape", observation.shape)
        obs_vector, self.obs_hidden_state = self.obs_model(observation, self.obs_hidden_state)  # (h_0, c_0)
        print("obs vector shape", obs_vector.shape)
        maxlen = max([len(item) for item in admissible_commands])
        command_logits = torch.zeros(len(admissible_commands), maxlen)
        for i, commands in enumerate(admissible_commands):
            prep_commands = torch.stack([self.prepare_string(command) for command in commands]).to(self.device)
            print("prep_command shape", prep_commands.shape)
            action_vector = self.act_model(prep_commands)
            print("action vector shape", action_vector.shape)
            # shapes of action_vector and obs_vector: [max(amount_of_actions), output_size], [output_size, batch_size]
            logit = torch.mean(torch.chain_matmul(action_vector, obs_vector.transpose(1, 0)), dim=1)
            command_logits[i] = torch.cat([logit, (-np.inf)*torch.ones(maxlen - logit.shape[0])]) if logit.shape[0] < maxlen \
                                                                                         else logit

        return command_logits

    def eval(self):
        """
        Method for evaluation. It would be necessary to implement this method if
        there will be any batch_norm / dropout layers in self.obs_model or self.act_model models.
        :return:
        """
        pass

    def act(self, states, infos):
        """
        in the case of batch_size state is [batch_size, state text] dimensional
        and info is still dict of several keys ("admissible commands", "verbs", "entities", etc.),
        but every value is batch-size length list, which elements are as usual
        :param obs: list of string by length of batch_size - observation received from the environment
        :param infos: environment additional information
        :return: actions - list of str - actions to play,
        """
        '''
        actions = []
        taken_action_probs = []
        for env in range(len(states)):
            admissible_commands = infos["admissible_commands"][env]
            command_logits = self.get_logits(states[env], admissible_commands, infos["extra.recipe"][env])
            command_probs = F.softmax(command_logits).to(self.device)
            action = np.random.choice(admissible_commands, p=command_probs.cpu().data.numpy())
            encoder = OneHotEncoder().fit([[i] for i in range(len(admissible_commands))])
            one_hot_index = encoder.transform([[admissible_commands.index(action)]]).toarray()
            one_hot_action = torch.FloatTensor(one_hot_index).squeeze().to(self.device)
            taken_action_prob = torch.dot(command_probs, one_hot_action)
            actions.append(action)
            taken_action_probs.append(taken_action_prob)
        
        return actions, torch.stack(taken_action_probs).to(self.device)
        '''
        admissible_commands = infos["admissible_commands"]  # list with the length of batch_size
        # command_logits.shape = (batch_size, max of admissible commands)
        command_logits = self.get_logits(states, admissible_commands, infos["extra.recipe"])
        command_probs = F.softmax(command_logits, dim=1)
        actions = [np.random.choice(admissible_commands[env],
                                    p=command_probs[env].cpu().data.numpy()[:len(admissible_commands[env])])
                   for env in range(self.batch_size)]
        one_hot_actions = torch.zeros(self.batch_size, command_probs.shape[1])
        for i in range(self.batch_size):
            j = admissible_commands[i].index(actions[i])
            one_hot_actions[i][j] = 1
        taken_actions_probs = torch.sum(command_probs*one_hot_actions, dim=1)

        return actions, taken_actions_probs

    def update(self, actions_probs, batch_states, batch_next_states, batch_rewards):
        """
        Updating agent parameters
        :param actions_probs: Tensor on self.device
        :param
        """
        batch_states = torch.stack([self.prepare_string(state) for state in batch_states]).to(self.device)
        # batch_next_states = np.array([self.prepare_string(state) for state in batch_next_states], dtype=np.float64)
        batch_next_states = torch.stack([self.prepare_string(state) for state in batch_next_states]).to(self.device)
        batch_rewards = torch.FloatTensor(batch_rewards)
        # critic loss
        obs_vector_v_s, self.obs_hidden_state = self.obs_model(torch.cat([batch_states, self.recipe_vectors], dim=1),
                                                               self.obs_hidden_state)
        v_s = self.state_value(obs_vector_v_s).squeeze()
        obs_vec_v_s_next, self.obs_hidden_state = self.obs_model(torch.cat([batch_next_states, self.recipe_vectors], dim=1),
                                                                 self.obs_hidden_state)
        v_s_next = self.state_value(obs_vec_v_s_next).squeeze()
        target_v_s = batch_rewards + self.gamma*v_s_next
        self.critic_loss = torch.mean((v_s - target_v_s.detach())**2)
        self.critic_loss.backward(retain_graph=True)
        self.critic_opt.step()
        self.critic_opt.zero_grad()
        # actor loss
        advantage = batch_rewards + self.gamma*v_s_next - v_s
        J = torch.mean(torch.log(actions_probs)*advantage.detach())
        H = -torch.mean(actions_probs*torch.log(actions_probs))
        self.actor_loss = -J - self.entropy_coef*H
        self.actor_loss.backward()
        self.act_opt.step()
        self.act_opt.zero_grad()

        return J.cpu().data.numpy(), H.cpu().data.numpy(), self.critic_loss.cpu().data.numpy()
