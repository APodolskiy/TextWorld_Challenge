import json
import random
from typing import Dict, List, Any

from _jsonnet import evaluate_file
import spacy
import torch
import torch.nn.functional as F

from textworld import EnvInfos

from agents.utils.generic import to_np, to_pt, preproc, _words_to_ids, pad_sequences, max_len
from agents.utils.eps_scheduler import LinearScheduler
from agents.baseline_dqn.model import LSTM_DQN
from agents.utils.replay import BinaryPrioritizeReplayMemory


class CustomAgent:
    def __init__(self):
        self.mode = "train"
        self.word2id = {}
        self.word_vocab = []
        self._load_vocab(vocab_file="./vocab.txt")
        self.EOS_id = self.word2id["</S>"]
        self.SEP_id = self.word2id["SEP"]

        self.config = json.loads(evaluate_file("configs/dqn_config.jsonnet"))

        self.nb_epochs = self.config['training']['nb_epochs']
        self.batch_size = self.config['training']['batch_size']
        self.max_nb_steps_per_episode = self.config['training']['max_nb_steps_per_episode']

        self.use_cuda = True and torch.cuda.is_available()
        self.model = LSTM_DQN(config=self.config["model"], word_vocab=self.word_vocab)
        self.action_head = None

        self.nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])

        self.eps_scheduler = LinearScheduler({})
        self.act_steps = 0

        self._episode_started = False
        self.previous_actions: List[str] = []
        self.scores: List[int] = []
        self.dones: List[int] = []

    def act(self, obs: List[str], scores: List[int], dones: List[bool], infos: Dict[str, List[Any]]) -> List[str]:
        if not self._episode_started:
            self._init(obs, infos)

        if self.mode == "eval":
            return self.act_eval(obs, scores, dones, infos)

        if self.current_step > 0:
            # append scores / dones from previous step into memory
            self.scores.append(scores)
            self.dones.append(dones)
            # compute previous step's rewards and masks
            rewards_np, rewards, mask_np, mask = self.compute_reward()

        admissible_commands = infos["admissible_commands"]
        # Choose actions
        actions = []
        if random.random() > self.eps_scheduler(self.act_steps):
            actions.extend([random.choice(env_commands) for env_commands in admissible_commands])
        else:
            with torch.no_grad():
                input_description, description_id_list = self.get_game_state_info(obs, infos)
                # TODO: implement act method for simple DQN model.
                preprocessed_commands = self.preprocess_commands(admissible_commands)
                for description, preprocessed_commands, command_texts \
                        in zip(input_description, preprocessed_commands, admissible_commands):
                    command_idx = self.get_command(description, preprocessed_commands)
                    actions.append(command_texts[command_idx])

        self.previous_actions = actions
        # Update experience replay memory
        pass

        # Update model
        pass

        if all(dones):
            # Nothing to return if all environments terminated
            return

        return actions

    def act_eval(self, obs: List[str], scores: List[int], dones: List[bool], infos: Dict[str, List[Any]]) -> List[str]:
        pass

    def update(self, transition):
        # Update network parameters
        pass

    def get_command(self, description, commands) -> int:
        q_values = self.model(description, commands)
        command_idx = q_values.argmax().item()
        return command_idx

    def get_game_state_info(self, obs: List[str], infos: [Dict[str, List[Any]]]):
        inventory_token_list = [preproc(item, tokenizer=self.nlp) for item in infos["inventory"]]
        inventory_id_list = [_words_to_ids(tokens, self.word2id) for tokens in inventory_token_list]

        feedback_token_list = [preproc(item, str_type="feedback", tokenizer=self.nlp) for item in obs]
        feedback_id_list = [_words_to_ids(tokens, self.word2id) for tokens in feedback_token_list]

        recipe_token_list = [preproc(item, tokenizer=self.nlp) for item in infos["extra.recipe"]]
        recipe_id_list = [_words_to_ids(tokens, self.word2id) for tokens in recipe_token_list]

        description_token_list = [preproc(item, tokenizer=self.nlp) for item in infos["description"]]
        for i, d in enumerate(description_token_list):
            if len(d) == 0:
                description_token_list[i] = ["end"]
        description_id_list = [_words_to_ids(tokens, self.word2id) for tokens in description_token_list]

        state_id_list = [d + [self.SEP_id] +
                         i + [self.SEP_id] + r
                         for (d, i, r) in zip(description_id_list, inventory_id_list, recipe_id_list)]
        input_description = pad_sequences(state_id_list, maxlen=max_len(state_id_list)).astype('int32')
        # TODO: re-write for Pytorch 1 style device assignment
        input_description = to_pt(input_description)
        return input_description, description_id_list

    def preprocess_commands(self, commands):
        preprocessed_commands = []
        for commands_list in commands:
            commands_tokens = [preproc(item, tokenizer=self.nlp) for item in commands_list]
            commands_ids = [_words_to_ids(tokens, self.word2id) for tokens in commands_tokens]
            commands_description = pad_sequences(commands_ids, maxlen=max_len(commands_ids)).astype('int32')
            commands_description = to_pt(commands_description)
            preprocessed_commands.append(commands_description)
        return preprocessed_commands

    def _init(self, obs: List[str], infos: Dict[str, List[Any]]) -> None:
        self._episode_started = True
        self.scores = []
        self.dones = []
        self.prev_actions = ["" for _ in range(len(obs))]

    def _load_vocab(self, vocab_file: str) -> None:
        with open(vocab_file, "r") as fp:
            self.word_vocab = fp.read().split("\n")
        for i, w in enumerate(self.word_vocab + ['SEP']):
            self.word2id[w] = i

    def train(self):
        """
        Tell the agent that it's training phase.
        """
        self.mode = "train"
        self.model.train()

    def eval(self):
        """
        Tell the agent that it's evaluation phase.
        """
        self.mode = "eval"
        self.model.eval()

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

            >>> from textworld import EnvInfos
            >>> request_infos = EnvInfos(description=True, inventory=True, extras=["recipe"])
            ...
            >>> env = gym.make(env_id)
            >>> ob, infos = env.reset()
            >>> print(infos["description"])
            >>> print(infos["inventory"])
            >>> print(infos["extra.recipe"])

        Notes:
            The following information *won't* be available at test time:

            * 'walkthrough'
        """
        request_infos = EnvInfos()
        request_infos.description = True
        request_infos.inventory = True
        request_infos.entities, request_infos.verbs = True, True
        request_infos.extras = ["recipe"]
        request_infos.admissible_commands = True
        return request_infos
