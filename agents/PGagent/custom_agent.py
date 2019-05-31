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
from agents.PGagent.network import FCNetwork
from os.path import realpath as path

import warnings
warnings.simplefilter('ignore')


class CustomAgent:
    def __init__(self):
        # agent configuration
        with open('./config.yaml') as config_file:
            self.config = yaml.safe_load(config_file)
        try:
            self.vector_size = int(self.config['training']['vector_size'])
            self.hidden_size = int(self.config['training']['hidden_size'])
            self.output_size = int(self.config['training']['output_size'])
            self.gamma = float(self.config['training']['gamma'])
            self.learning_rate = float(self.config['training']['learning_rate'])
            self.batch_size = int(self.config['training']['batch_size'])
            self.max_nb_steps_per_episode = int(self.config['training']['max_nb_steps_per_episode'])
            self.entropy_coef = float(self.config['training']['entropy_coefficient'])
            self.device = str(self.config['training']['device'])
        except KeyError:
            print("Check and double check config.yaml file for typos and errors, parameters will be set to default")
            # set default parameters
            self.vector_size = 50
            self.hidden_size = 200
            self.output_size = 100
            self.gamma = 0.99
            self.learning_rate = 1e-4
            self.batch_size = None
            self.max_nb_steps_per_episode = 50
            self.entropy_coef = 0.1
            self.device = 'cpu'

        self.params = {'vector_size': self.vector_size,
                       'hidden_size': self.hidden_size,
                       'output_size': self.output_size,
                       'gamma': self.gamma,
                       'self.learning_rate':  self.learning_rate,
                       'batch_size': self.batch_size,
                       'max_nb_steps_per_episode': self.max_nb_steps_per_episode,
                       'entropy_coef': self.entropy_coef,
                       'self.device': self.device
        }

        with open("../../vocab.txt", 'r') as file:
            vocab = file.read().split('\n')

        print('[AGENT INFO] Loading word vectors')
        # word vectors: 08.04 -> glove 50d | 07.05 -> fasttext wiki-news-300d-1M.vec
        # for my laptop the path is /media/nik/hdd-data/datasets/glove/
        # and for my work pc the path is /home/nik-96/Documents/datasets/glove/
        # with open("/media/nik/hdd-data/datasets/glove/glove.6B.{}d.txt".format(self.vector_size)) as file:
        # 19.05 see transform_word_vectors.py, where it's formed word_vectors file from vocab.txt file
        with open('./fasttext_word_vectors.vec') as file:
            #                     word            :      vector
            self.wordVectors = {word_vector.split(' ')[0]: np.array(word_vector.split(' ')[1:], dtype='float')
                                for word_vector in file.read().split('\n')}
            self.wordVectors.pop('')
        print('[AGENT INFO] Word vectors were loaded')
        # word: id
        self.vocab = {word: i for i, word in enumerate(vocab)}

        self.trans_table = str.maketrans('\n', ' ', String.punctuation)
        self.StringVectors = {}

        # a little more serious model, taking last 4 observation
        self.obs_model = FCNetwork(4*self.vector_size, self.hidden_size, self.output_size, self.device).to(self.device)
        self.act_model = FCNetwork(self.vector_size, self.hidden_size, self.output_size, self.device).to(self.device)

        self.optimizer = torch.optim.Adam(list(self.obs_model.parameters()) + list(self.act_model.parameters()),
                                          lr=self.learning_rate)

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
        request_infos.description = True
        request_infos.has_won = True
        request_infos.has_lost = True
        request_infos.inventory = True
        request_infos.entities = True
        request_infos.verbs = True
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

            obs_state_dict = torch.load(load_from + '/obs_model.pt')
            act_state_dict = torch.load(load_from + '/act_model.pt')
            self.obs_model.load_state_dict(obs_state_dict)
            self.act_model.load_state_dict(act_state_dict)
        except:
            print("[INFO] Failed to load checkpoint...")

    def save_model(self, save_path: str):
        """
        Save checkpoint to file.
        :param save_path: path to directory (without '/' in the end), where to save models
        """
        print(f"\n[INFO] Saving model to {path(save_path) + '/'}")
        try:
            torch.save(self.obs_model.state_dict(), save_path + f'/obs_model.pt')
            torch.save(self.act_model.state_dict(), save_path + f'/act_model.pt')
        except:
            print("[INFO] Failed to save checkpoint...")

    def prepare_string(self, string):
        """
        This function preprocesses string feedback from environment and return torch tensor of average token embeddings
        :param string: str feedback from environment
        :return: torch.tensor
        """
        # TODO: add string cashing
        string_vector_ = self.StringVectors.get(string, None)
        if string_vector_ is not None:
            return torch.FloatTensor(string_vector_)
        prep_string = re.split(' ', re.sub('[ ]+', ' ', string.lower().translate(self.trans_table)))
        string_vector = np.array([self.wordVectors[word] for word in prep_string
                                  if word in self.wordVectors and word not in STOP_WORDS])
        string_vector = np.mean(string_vector, axis=0)
        self.StringVectors[string] = string_vector

        return torch.FloatTensor(string_vector)

    def get_cumulative_rewards(self, rewards):
        """
        take a list of immediate rewards r(s,a) for the whole session
        compute cumulative returns (a.k.a. G(s,a) in Sutton '16)
        G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

        The simple way to compute cumulative rewards is
        to iterate from last to first time tick
        and compute G_t = r_t + gamma*G_{t+1} recurrently

        You must return an array/list of cumulative rewards with as many elements
        as in the initial rewards.
        """
        def G_t(reward_arr, gamma):
            return sum([gamma ** index * r for index, r in enumerate(reward_arr)])

        G = [G_t(rewards[index:], self.gamma) for index, r in enumerate(rewards)]

        return np.array(G)

    def get_logits(self, state, admissible_commands, recipe):
        """
        Getting logits from two neural networks
        :param state: string - observation from environment
        :param admissible_commands: list of strings - possible commands in state
        :param recipe: string - recipe description
        :return: list of logits from networks, length of returned list equals to length of admissible_commands list
        """
        # TODO: move prepared string to self.device
        prep_obs = torch.stack([torch.zeros(self.vector_size) if item == '' else self.prepare_string(item)
                                for item in state])
        prep_obs = prep_obs.reshape(-1).to(self.device)
        obs_vector = self.obs_model(prep_obs)
        recipe_vector = self.prepare_string(recipe).to(self.device)
        command_logits = torch.zeros(len(admissible_commands))
        for i, command in enumerate(admissible_commands):
            prep_command = self.prepare_string(command).to(self.device)
            action_vector = self.act_model(prep_command)
            command_logits[i] = torch.dot(action_vector, obs_vector) + torch.dot(prep_command, recipe_vector)

        return command_logits

    def act(self, states, infos):
        """
        in the case of batch_size state is [batch_size, state text] dimensional
        and info is still dict of several keys ("admissible commands", "verbs", "entities", etc.),
        but every value is batch-size length list, which elements are as usual
        :param obs: list of string by length of batch_size - observation received from the environment
        :param infos: environment additional information
        :return: actions - list of str - actions to play,
        """
        actions = []
        taken_action_probs = []
        states = [[i1, i2, i3, i4] for i1, i2, i3, i4 in zip(*states)]
        for env in range(len(states)):
            admissible_commands = infos["admissible_commands"][env]
            command_logits = self.get_logits(states[env], admissible_commands, infos["extra.recipe"][env])
            command_probs = F.softmax(command_logits).to(self.device)
            action = np.random.choice(admissible_commands, p=command_probs.cpu().data.numpy())
            encoder = OneHotEncoder()
            encoder.fit([[i] for i in range(len(admissible_commands))])
            index = admissible_commands.index(action)
            one_hot_index = encoder.transform([[index]]).toarray()
            one_hot_action = torch.FloatTensor(one_hot_index).squeeze().to(self.device)
            taken_action_prob = torch.dot(command_probs, one_hot_action)
            actions.append(action)
            taken_action_probs.append(taken_action_prob)

        return actions, torch.stack(taken_action_probs).to(self.device)

    def update(self, actions_probs, batch_rewards):
        """
        Updating agent parameters
        :param actions_probs: [len(episode), batch_size]
        :param batch_rewards: [len(episode), batch_size]
        :return: loss and entropy
        """
        # cumulative_rewards is 2d array: episode*len(episode)
        batch_rewards = nn.utils.rnn.pad_sequence([torch.FloatTensor(r) for r in batch_rewards])
        # now batch_rewards.shape = [batch_size, len(episode)]
        batch_rewards = np.array([self.get_cumulative_rewards(episode_rewards)
                                  for episode_rewards in batch_rewards])
        cumulative_rewards = torch.FloatTensor(batch_rewards).to(self.device)
        actions_probs = torch.nn.utils.rnn.pad_sequence(actions_probs)
        entropy = -torch.mean(actions_probs*torch.log(actions_probs))
        J = torch.mean(torch.log(actions_probs)*torch.sum(cumulative_rewards))
        self.loss = -J - self.entropy_coef*entropy
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return self.loss.cpu().data.numpy(), entropy.cpu().data.numpy()
