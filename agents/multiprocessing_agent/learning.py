from logging import info, warning
from multiprocessing import Queue
from time import sleep
from typing import Optional

from numpy import array
from tensorboardX import SummaryWriter
import torch
from torch.nn.functional import smooth_l1_loss
from torch.optim import Adam

from agents.multiprocessing_agent.custom_agent import Transition, QNet
from agents.utils.replay import AbstractReplayMemory, BinaryPrioritizeReplayMemory


def learn(
    policy_net: QNet,
    target_net: QNet,
    replay_buffer: BinaryPrioritizeReplayMemory,
    queue: Queue,
    params,
    log_dir: Optional[str],
):
    if log_dir is not None:
        writer = SummaryWriter(log_dir)
    info("Started learning process")
    max_samples = params.pop("max_samples")
    batch_size = params.pop("batch_size")
    gamma = params.pop("gamma")
    optimizer = Adam(policy_net.parameters(), lr=params.pop("lr"))
    for learning_step in range(params.pop("n_learning_steps")):
        samples = 0
        while not queue.empty() and samples < max_samples:
            samples += 1
            transition = queue.get()
            replay_buffer.push(transition, is_prior=transition.reward != 0)

        try:
            batch = Transition(*zip(*replay_buffer.sample(batch_size)))
            policy_net.train()
            q_values_selected_actions = torch.cat(
                policy_net(batch.previous_state, batch.action)
            )

            non_terminal_idxs = ~array(batch.done)
            next_state_values = torch.zeros(len(batch.reward), device=policy_net.device)

            next_non_final_states = array(batch.next_state)[non_terminal_idxs]
            next_non_final_allowed_actions = array(batch.allowed_actions)[
                non_terminal_idxs
            ]
            target_net.eval()
            next_state_q_values = target_net(
                next_non_final_states, next_non_final_allowed_actions
            )

            tensor_indices = torch.tensor(
                tuple(non_terminal_idxs), dtype=torch.uint8, device=policy_net.device
            )

            next_state_values[tensor_indices] = torch.tensor(
                [q_values.max().item() for q_values in next_state_q_values],
                device=policy_net.device,
            )
            expected_values = (
                torch.tensor(batch.reward, device=q_values_selected_actions.device)
                + gamma * next_state_values
            )

            optimizer.zero_grad()
            loss = smooth_l1_loss(q_values_selected_actions, expected_values)
            loss.backward()
            optimizer.step()
            if log_dir is not None:
                writer.add_scalar("train/loss", loss.item(), learning_step)
                writer.add_histogram(
                    "train/learner_target_net_weights",
                    target_net.hidden_to_scores.weight.clone().detach().cpu().numpy(),
                    learning_step,
                )
                writer.add_histogram(
                    "train/learner_policy_net_weights",
                    policy_net.hidden_to_scores.weight.clone().detach().cpu().numpy(),
                    learning_step,
                )

            info(f"Done learning step, loss={loss.item()}")

        except ValueError:
            warning("Not enough elements in buffer!")
            sleep(2.0)
