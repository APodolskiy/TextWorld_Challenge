from logging import info, warning
from multiprocessing import Queue
from time import sleep
from tensorboardX import SummaryWriter
import torch
from torch.nn.functional import smooth_l1_loss
from torch.optim import Adam

from agents.multiprocessing_agent.custom_agent import Transition, QNet
from agents.utils.replay import AbstractReplayMemory


def learn(
    policy_net: QNet,
    target_net: QNet,
    replay_buffer: AbstractReplayMemory,
    queue: Queue,
    params,
    log_dir: str
):
    sleep(2.0)
    writer = SummaryWriter()
    info("Started learning process")
    max_samples = params.pop("max_samples")
    batch_size = params.pop("batch_size")
    gamma = params.pop("gamma")
    optimizer = Adam(policy_net.parameters(), lr=params.pop("lr"))
    for learning_step in range(params.pop("n_learning_steps")):
        samples = 0
        while not queue.empty() and samples < max_samples:
            samples += 1
            replay_buffer.push(queue.get())

        try:
            batch = Transition(*zip(*replay_buffer.sample(batch_size)))
            q_values_selected_actions = torch.cat(
                policy_net(batch.previous_state, batch.action)
            )

            next_state_q_values = target_net(batch.next_state, batch.allowed_actions)
            next_state_values = torch.tensor(
                [q_values.max().item() for q_values in next_state_q_values],
                device=policy_net.device,
            )
            # TODO: terminal states?
            expected_values = (
                torch.tensor(batch.reward, device=q_values_selected_actions.device)
                + gamma * next_state_values
            )

            optimizer.zero_grad()
            loss = smooth_l1_loss(q_values_selected_actions, expected_values)
            loss.backward()
            optimizer.step()

            writer.add_scalar("train/loss", loss.item(), learning_step)
            info(f"Done learning step, loss={loss.item()}")

        except ValueError:
            warning("Not enough elements in buffer!")
            sleep(2.0)
