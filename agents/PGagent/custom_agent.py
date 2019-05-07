import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import textworld
import nltk
import yaml
from textworld import EnvInfos
from agents.PGagent.network import FCNetwork


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
            self.obs_learning_rate = float(self.config['training']['obs_learning_rate'])
            self.act_learning_rate = float(self.config['training']['act_learning_rate'])
            self.batch_size = int(self.config['training']['batch_size'])
            self.max_nb_steps_per_episode = int(self.config['training']['max_nb_steps_per_episode'])
            self.entropy_coef = float(self.config['training']['entropy_coefficient'])
        except KeyError:
            print("Check and double check config.yaml file for typos and errors, parameters will be set to default")
            # set default parameters
            self.vector_size = 50
            self.hidden_size = 200
            self.output_size = 100
            self.gamma = 0.99
            self.obs_learning_rate = 1e-3
            self.act_learning_rate = 1e-3
            self.batch_size = None
            self.max_nb_steps_per_episode = 50
            self.entropy_coef = 0.1

        with open("../../vocab.txt", 'r') as file:
            vocab = file.read().split('\n')

        # word vectors: 08.04 -> glove 50d | 07.05 -> fasttext wiki-news-300d-1M.vec
        # for my laptop the path is /media/nik/hdd-data/datasets/glove/
        # and for my work pc the path is /home/nik-96/Documents/datasets/glove/
        # with open("/home/nik-96/Documents/datasets/glove/glove.6B.{}d.txt".format(self.vector_size)) as file:
        with open('/home/nik-96/Documents/datasets/fasttext/wiki-news-300d-1M.vec') as file:
            #                     word            :      vector
            self.wordVectors = {item.split(' ')[0]: np.array(item.split(' ')[1:], dtype='float')
                                for item in file.read().split('\n')}

        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        # word: id
        self.vocab = {word: i for i, word in enumerate(vocab)}

        self.tokenizer = nltk.tokenize.WordPunctTokenizer()
        self.stemmer = nltk.stem.PorterStemmer()

        # simple model, taking just observation
        self.obs_model = FCNetwork(self.vector_size, self.hidden_size, self.output_size)
        self.act_model = FCNetwork(self.vector_size, self.hidden_size, self.output_size)

        self.obs_optimizer = torch.optim.Adam(self.obs_model.parameters(), lr=self.obs_learning_rate)
        self.act_optimizer = torch.optim.Adam(self.act_model.parameters(), lr=self.act_learning_rate)

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
        request_infos.inventory = True
        request_infos.entities = True
        request_infos.verbs = True
        request_infos.admissible_commands = True
        request_infos.extras = ["recipe"]
        return request_infos

    def load_pretrained_model(self, load_from):
        """
        Load pretrained checkpoint from file.

        Arguments:
            load_from: path to directory (without '/' in the end), that contains obs model and act model
        """
        print("[INFO] Loading model from %s\n" % load_from)
        try:

            obs_state_dict = torch.load(load_from + '/obs_model.pt')
            act_state_dict = torch.load(load_from + '/act_model.pt')
            self.obs_model.load_state_dict(obs_state_dict)
            self.act_model.load_state_dict(act_state_dict)
        except:
            print("[INFO] Failed to load checkpoint...")

    def save_model(self, save_path):
        """

        :param save_path:  path to directory (without '/' in the end), where to save models
        """
        print(f"[INFO] Saving model to {save_path}\n")
        try:
            torch.save(self.obs_model.state_dict(), save_path + '/obs_model.pt')
            torch.save(self.act_model.state_dict(), save_path + '/act_model.pt')
        except:
            print("[INFO] Failed to save checkpoint...")

    def prepare_string(self, string):
        string = self.tokenizer.tokenize(string)
        string = [word for word in string if word not in self.stopwords]
        string_vector = np.array([self.wordVectors[word] for word in string if word in self.wordVectors.keys()])
        string_vector = np.mean(string_vector, axis=0)

        return string_vector

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

    def get_logits(self, state, admissible_commands):
        """
        Getting logits from two neural networks
        :param state: string - observation from environment
        :param admissible_commands: list of strings - possible commands in state
        :return: list of logits from networks, length of returned list equals to length of admissible_commands list
        """
        prep_obs = self.prepare_string(state)
        obs_vector = self.obs_model(prep_obs)
        command_logits = torch.zeros(len(admissible_commands))
        for i, command in enumerate(admissible_commands):
            prep_command = self.prepare_string(command)
            action_vector = self.act_model(prep_command)
            command_logits[i] = torch.dot(action_vector, obs_vector)

        return command_logits

    def act(self, states, infos):
        """
        in the case of batch_size state is [batch_size, state text] dimensional
        and info is still dict of several keys ("admissible commands", "verbs", "entities", etc.),
        but every value is batch-size length list, which elements are as usual

        :param obs: list of string by length of batch_size - observation received from the environment
        :param infos: environment additional information
        :return: action to play
        """
        actions = []
        taken_action_probs = []
        for env in range(len(states)):
            admissible_commands = infos["admissible_commands"][env]
            command_logits = self.get_logits(states[env], admissible_commands)
            command_probs = F.softmax(command_logits)
            action = np.random.choice(admissible_commands, p=command_probs.data.numpy())
            taken_action_prob = command_probs[admissible_commands.index(action)]

            actions.append(action)
            taken_action_probs.append(taken_action_prob)

        return actions, np.array(taken_action_probs)

    def update(self, actions_probs, rewards):
        """
        Updating agent parameters
        :param action_probs: [len(episode), batch_size]
        :param rewards: [len(episode), batch_size]
        :return: None
        """
        # cumulative_rewards is 2d array: episode*len(episode)
        # TODO: add padding for episode rewards
        # here is bug about episode length - is some episode ends quicker than self.max_nb_steps_per_episode
        # then there will be different length vectors consisting matrix - BUG :)
        cumulative_rewards = torch.FloatTensor(np.array([self.get_cumulative_rewards(episode_rewards)
                                                         for episode_rewards in rewards]))
        actions_probs = torch.tensor(actions_probs.astype('float'), dtype=torch.float32)
        actions_probs = torch.autograd.Variable(actions_probs, requires_grad=True)
        entropy = -torch.mean(actions_probs*torch.log(actions_probs))
        J = torch.mean(torch.log(actions_probs)*cumulative_rewards)
        self.loss = - J - self.entropy_coef*entropy
        self.loss.backward()
        self.obs_optimizer.step()
        self.act_optimizer.step()
        self.obs_optimizer.zero_grad()
        self.act_optimizer.zero_grad()

        return self.loss.data.numpy(), entropy.data.numpy()
