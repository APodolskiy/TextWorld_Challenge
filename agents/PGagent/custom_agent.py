import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import textworld
import nltk
from textworld import EnvInfos
from agents.PGagent.network import Network


class CustomAgent:
    def __init__(self):
        self.vector_size = 50
        self.hidden_size = 200
        self.output_size = 100
        self.gamma = 0.99
        self.learning_rate = 0.001
        with open("../../vocab.txt", 'r') as file:
            vocab = file.read().split('\n')

        # word vectors: 08.04 -> glove 50d
        with open("/home/nik-96/Documents/datasets/glove/glove.6B.{}d.txt".format(self.vector_size)) as file:
            #                     word            :      vector
            self.wordVectors = {item.split(' ')[0]: np.array(item.split(' ')[1:], dtype='float')
                                for item in file.read().split('\n')}

        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        # word: id
        self.vocab = {word: i for i, word in enumerate(vocab)}

        self.tokenizer = nltk.tokenize.WordPunctTokenizer()
        self.stemmer = nltk.stem.PorterStemmer()

        # simple model, taking just observation
        self.obs_model = Network(self.vector_size, self.hidden_size, self.output_size)
        self.act_model = Network(self.vector_size, self.hidden_size, self.output_size)

        self.optimizer = torch.optim.Adam(list(self.obs_model.parameters()) + \
                                          list(self.act_model.parameters()), lr=self.learning_rate)

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
        request_infos.extras = ["recipe"]
        return request_infos

    def load_pretrained_model(self, load_from):
        """
        Load pretrained checkpoint from file.

        Arguments:
            load_from: path to directory (without '/' in the end), that contains obs model and act model
        """
        print("loading model from %s\n" % load_from)
        try:

            obs_state_dict = torch.load(load_from + '/obs_model.pt')
            act_state_dict = torch.load()
            self.obs_model.load_state_dict(obs_state_dict)
            self.act_model.load_state_dict(act_state_dict)
        except:
            print("Failed to load checkpoint...")

    def save_model(self, save_path):
        """

        :param save_path:  path to directory (without '/' in the end), where to save models
        """
        print(f"saving model to {save_path}\n")
        try:
            torch.save(self.obs_model.state_dict(), save_path + '/obs_model.pt')
            torch.save(self.act_model.state_dict(), save_path + '/act_model.pt')
        except:
            print("Failed to save checkpoint...")

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

    def get_logits(self, state, info):
        prep_obs = self.prepare_string(state)
        obs_vector = self.obs_model(prep_obs)  # {self.output_size}-d vector
        admissible_commands = info['admissible_commands']
        command_logits = torch.zeros(len(admissible_commands))
        for i, command in enumerate(admissible_commands):
            prep_command = self.prepare_string(command)
            action_vector = self.act_model(prep_command)
            command_logits[i] = torch.dot(action_vector, obs_vector)

        return admissible_commands, command_logits

    def act(self, state, info):
        """
        :param obs: string - observation received from the environment
        :param infos: environment additional information
        :return: action to play
        """
        admissible_commands, command_logits = self.get_logits(state, info)

        command_probs = F.softmax(command_logits)
        action = np.random.choice(admissible_commands, p=command_probs.data.numpy())
        taken_action_prob = command_probs[admissible_commands.index(action)]

        return action, taken_action_prob

    def update(self, action_probs, rewards):
        """
        Updating agent parameters
        :param action_probs: [len(episode)]
        :param rewards: [len(episode)]
        :return: None
        """
        # cumulative_rewards is 2d array: episode*len(episode)
        cumulative_rewards = torch.FloatTensor(self.get_cumulative_rewards(rewards))
        action_probs = torch.tensor(action_probs.astype('float'), dtype=torch.float32)
        action_probs = torch.autograd.Variable(action_probs, requires_grad=True)
        entropy = torch.mean(action_probs*torch.log(action_probs))
        J = torch.mean(torch.log(action_probs)*cumulative_rewards)
        self.loss = - J - 0.1*entropy
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return self.loss.data.numpy(), entropy.data.numpy()
