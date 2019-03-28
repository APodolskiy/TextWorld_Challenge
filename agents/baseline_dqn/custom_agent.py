from collections import namedtuple
import json
import random
from typing import Dict, List, Any

from _jsonnet import evaluate_file
import numpy as np
import spacy
import torch
import torch.optim as optim
import torch.nn.functional as F
from spacy.lang.en import STOP_WORDS
from tensorboardX import SummaryWriter

from textworld import EnvInfos

from agents.utils.generic import to_np, to_pt, preproc, _words_to_ids, pad_sequences, max_len
from agents.utils.eps_scheduler import LinearScheduler
from agents.baseline_dqn.model import LSTM_DQN
from agents.utils.replay import BinaryPrioritizeReplayMemory, TernaryPrioritizeReplayMemory


Transition = namedtuple('Transition', ('description_id_list', 'command',
                                       'reward', 'mask', 'done',
                                       'next_description_id_list',
                                       'next_commands'))


class CustomAgent:
    def __init__(self, writer: SummaryWriter):
        self.mode = "train"
        self.word2id = {}
        self.word_vocab = []
        self._load_vocab(vocab_file="./vocab.txt")
        self.EOS_id = self.word2id["</S>"]
        self.SEP_id = self.word2id["SEP"]

        self.writer = writer

        self.config = json.loads(evaluate_file("configs/dqn_config.jsonnet"))

        self.nb_epochs = self.config['training']['nb_epochs']
        self.batch_size = self.config['training']['batch_size']
        self.max_nb_steps_per_episode = self.config['training']['max_nb_steps_per_episode']

        self.use_cuda = True and torch.cuda.is_available()
        self.model = LSTM_DQN(config=self.config["model"], word_vocab=self.word_vocab)
        self.target_model = LSTM_DQN(config=self.config["model"], word_vocab=self.word_vocab)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        if self.use_cuda:
            self.model.cuda()
            self.target_model.cuda()

        self.replay_config = self.config["replay"]
        self.replay_memory = TernaryPrioritizeReplayMemory(capacity=self.replay_config["capacity"],
                                                           priority_fraction=self.replay_config["priority_fraction"])

        self.nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])

        self.update_per_k_game_steps = self.config["training"]["update_freq"]
        self.update_target = self.config["training"]["target_net_update_freq"]
        self.replay_batch_size = self.config["training"]["replay_batch_size"]
        self.clip_grad_norm = self.config["training"]["clip_grad_norm"]
        self.discount_gamma = self.config["training"]["discount_gamma"]

        self.eps_scheduler = LinearScheduler(self.config["exploration"])
        self.act_steps = 0
        self.episode_steps = 0
        self.num_episodes = 0

        self._episode_started = False
        self.previous_actions: List[str] = []
        self.scores: List[List[int]] = []
        self.dones: List[List[int]] = []
        self.prev_description_id: List = None
        self.prev_command: List = None

    def act(self, obs: List[str], scores: List[int], dones: List[bool], infos: Dict[str, List[Any]]) -> List[str]:
        if not self._episode_started:
            self._init(obs, infos)
            self._episode_started = True

        if self.mode == "eval":
            return self.act_eval(obs, scores, dones, infos)

        if self.episode_steps > 0:
            # append scores / dones from previous step into memory
            max_score = infos["max_score"]
            self.scores.append(scores)
            self.dones.append(dones)
            # compute previous step's rewards and masks
            rewards_np, rewards, mask_np, mask = self.compute_reward(obs, max_score)

        admissible_commands = infos["admissible_commands"]

        # Tokenize info
        input_description, description_id = self.get_game_state_info(obs, infos)
        input_commands, preprocessed_commands = self.preprocess_commands(admissible_commands)

        # Choose actions
        actions = []
        choosen_command_idx = []
        if random.random() < self.eps_scheduler(self.act_steps):
           choosen_command_idx.extend([np.random.choice(len(commands)) for commands in admissible_commands])
           actions.extend([env_commands[command_id]
                           for env_commands, command_id in zip(admissible_commands, choosen_command_idx)])
        else:
            with torch.no_grad():
                command_ids = self.get_commands(input_description, input_commands)
                choosen_command_idx.extend(command_ids)
                actions.extend(ad_cmd[cmd_id] for ad_cmd, cmd_id in zip(admissible_commands, choosen_command_idx))

        self.previous_actions = actions
        # Update experience replay memory
        if self.episode_steps > 0:
            for b in range(len(obs)):
                if mask_np[b] == 0:
                    continue
                transition = Transition(self.prev_description_id[b],
                                        self.prev_command[b],
                                        rewards[b],
                                        mask[b],
                                        dones[b],
                                        description_id[b],
                                        preprocessed_commands[b]
                                        )
                self.replay_memory.push(transition=transition)

        # Update model
        if self.episode_steps > 0 and self.episode_steps % self.update_per_k_game_steps == 0:
            loss = self.update()
            if loss is not None:
                # Backpropagate
                self.optimizer.zero_grad()
                loss.backward()
                self.writer.add_scalar("loss", loss.item(), self.act_steps)
                #loss.backward(retain_graph=True)
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()  # apply gradients

        if self.episode_steps > 0 and self.episode_steps % self.update_target == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        self.prev_description_id = description_id
        self.prev_command = [[preprocessed_commands[env_id][idx]] for env_id, idx in enumerate(choosen_command_idx)]
        self.episode_steps += 1
        self.act_steps += 1

        if all(dones):
            # Nothing to return if all environments terminated
            self.finish()
            self._episode_started = False
            return

        return actions

    def act_eval(self, obs: List[str], scores: List[int], dones: List[bool], infos: Dict[str, List[Any]]) -> List[str]:
        pass

    def update(self):
        # Update network parameters
        if len(self.replay_memory) < self.replay_batch_size:
            return None
        transitions = self.replay_memory.sample(batch_size=self.replay_batch_size)
        batch = Transition(*zip(*transitions))
        # Compute loss
        description_id = pad_sequences(batch.description_id_list,
                                       maxlen=max_len(batch.description_id_list)).astype('int32')
        input_description = to_pt(description_id, self.use_cuda)
        preprocessed_commands = [pad_sequences(commands, maxlen=max_len(commands)).astype('int32')
                                 for commands in batch.command]
        input_commands = [to_pt(prep_cmd, self.use_cuda) for prep_cmd in preprocessed_commands]
        q_value = self.model(input_description, input_commands)
        q_value = torch.stack(q_value, dim=0).squeeze(1)

        next_description_id = pad_sequences(batch.next_description_id_list,
                                            maxlen=max_len(batch.next_description_id_list)).astype('int32')
        next_input_description = to_pt(next_description_id, self.use_cuda)
        preprocessed_commands = [pad_sequences(commands, maxlen=max_len(commands)).astype('int32')
                                 for commands in batch.next_commands]
        input_commands = [to_pt(prep_cmd, self.use_cuda) for prep_cmd in preprocessed_commands]
        next_q_values_target = self.target_model(next_input_description, input_commands)
        next_q_values_target = [q_vals.detach() for q_vals in next_q_values_target]
        next_q_values_model = self.model(next_input_description, input_commands)
        next_q_values_model = [q_vals.detach() for q_vals in next_q_values_model]
        next_q_value = [target_q_vals[q_vals.argmax()]
                        for q_vals, target_q_vals in zip(next_q_values_model, next_q_values_target)]
        next_q_value = torch.stack(next_q_value, dim=0)

        rewards = torch.stack(batch.reward)  # batch
        not_done = 1.0 - np.array(batch.done, dtype='float32')  # batch
        not_done = to_pt(not_done, self.use_cuda, type='float')
        rewards = rewards + not_done * next_q_value * self.discount_gamma  # batch
        mask = torch.stack(batch.mask)
        loss = F.smooth_l1_loss(q_value * mask, rewards * mask)

        #print(f"Q-values: {q_value.detach().cpu().numpy()[::8]}\nRewards: {rewards.cpu().numpy()[::8]}")

        return loss

    def compute_reward(self, obs: List[str], max_score: List[int]):
        """
        Compute rewards by agent. Note this is different from what the training/evaluation
        scripts do. Agent keeps track of scores and other game information for training purpose.

        """
        # mask = 1 if game is not finished or just finished at current step
        if len(self.dones) == 1:
            # it's not possible to finish a game at 0th step
            mask = [1.0 for _ in self.dones[-1]]
        else:
            assert len(self.dones) > 1
            mask = [1.0 if not self.dones[-2][i] else 0.0 for i in range(len(self.dones[-1]))]
        mask = np.array(mask, dtype='float32')
        mask_pt = to_pt(mask, self.use_cuda, type='float')
        # rewards returned by game engine are always accumulated value the
        # agent have received. so the reward it gets in the current game step
        # is the new value minus values at previous step.
        rewards = np.array(self.scores[-1], dtype='float32')  # batch
        if len(self.scores) > 1:
            prev_rewards = np.array(self.scores[-2], dtype='float32')
            rewards = rewards - prev_rewards
            rewards += [3 if s == max_s else 0 for s, max_s in zip(self.scores[-1], max_score)]
        rewards += [-2 if 'lost' in f else 0 for f in obs]
        rewards_pt = to_pt(rewards, self.use_cuda, type='float')
        return rewards, rewards_pt, mask, mask_pt

    def get_commands(self, description, commands) -> List[int]:
        q_values = self.model(description, commands)
        commands_ids = [q_vs.argmax().item() for q_vs in q_values]
        # commands_ids = []
        # Boltzmann exploration
        # for q_vs in q_values:
        #     norm_q_vs: torch.Tensor = torch.nn.functional.softmax(q_vs / 0.1)
        #     commands_ids.append(int(norm_q_vs.multinomial(num_samples=1).item()))
        return commands_ids

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
        input_description = to_pt(input_description, self.use_cuda)
        return input_description, description_id_list

    def preprocess_commands(self, commands):
        preprocessed_commands = []
        input_commands = []
        for commands_list in commands:
            commands_tokens = [preproc(item, tokenizer=self.nlp) for item in commands_list]
            commands_ids = [_words_to_ids(tokens, self.word2id) for tokens in commands_tokens]
            preprocessed_commands.append(commands_ids)
            commands_description = pad_sequences(commands_ids, maxlen=max_len(commands_ids)).astype('int32')
            commands_description = to_pt(commands_description, self.use_cuda)
            input_commands.append(commands_description)
        return input_commands, preprocessed_commands

    def _init(self, obs: List[str], infos: Dict[str, List[Any]]) -> None:
        self._episode_started = True
        self.scores = []
        self.dones = []
        self.prev_command = ["" for _ in range(len(obs))]
        self.episode_steps = 0

    def finish(self):
        # TODO: update epsilon here
        self.num_episodes += 1

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
        request_infos.max_score = True
        request_infos.extras = ["recipe"]
        request_infos.admissible_commands = True
        return request_infos
