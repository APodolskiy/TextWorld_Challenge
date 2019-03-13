from multiprocessing import Queue

import torch
from torch.nn.functional import smooth_l1_loss
from torch.optim import Adam

from agents.multiprocessing_agent.custom_agent import Transition, QNet
from agents.utils.replay import AbstractReplayMemory


def learn(
    net: QNet, target_net: QNet, replay_buffer: AbstractReplayMemory, queue: Queue, params
):
    # TODO: put lr into config
    optimizer = Adam(net.parameters(), lr=params.pop("lr"))
    while not queue.empty():
        replay_buffer.push(queue.get())
    batch = Transition(*zip(*replay_buffer.sample(params.pop("batch_size"))))
    q_values_selected_actions = torch.cat(net(batch.previous_state, batch.action))

    next_state_q_values = target_net(batch.next_state, batch.allowed_actions)
    next_state_values = torch.tensor(
        [q_values.max().item() for q_values in next_state_q_values],
        device=next_state_q_values[0].device,
    )

    # TODO: put gamma in config
    # TODO: rewards are computed incorrectly: you need differences
    expected_values = q_values_selected_actions + params.pop("gamma") * torch.tensor(
        batch.reward, device=q_values_selected_actions.device,
    )
    optimizer.zero_grad()
    loss = smooth_l1_loss(next_state_values, expected_values)
    loss.backward()
    optimizer.step()
