from logging import info, warning
from multiprocessing import Queue
from queue import Empty
from time import sleep
from typing import Optional

from numpy import array
from tensorboardX import SummaryWriter
import torch
from torch.nn.functional import smooth_l1_loss
from torch.optim import Adam

from agents.utils.split import get_chunks
from agents.utils.types import Transition
from agents.utils.replay import SeqTernaryPrioritizeReplayMemory
from agents.utils.utils import idx_select

learning_step = 1


def learn(
    policy_net,
    target_net,
    replay_buffer: SeqTernaryPrioritizeReplayMemory,
    queue: Queue,
    params,
    log_dir: Optional[str],
):
    global learning_step
    if log_dir is not None:
        writer = SummaryWriter(log_dir)
    info("Started learning process")
    max_samples = params.pop("max_samples")
    batch_size = params.pop("batch_size")
    gamma = params.pop("gamma")
    saving_freq = params.pop("saving_freq")
    model_path = params.pop("model_path")
    optimizer = Adam(policy_net.parameters(), lr=params.pop("lr"))
    chunk_size = params.pop("chunk_size")
    for _ in range(params.pop("n_learning_steps")):
        samples = 0
        while samples < max_samples:
            try:
                transitions = queue.get(True, 1.0)
                for chunk in get_chunks(transitions, chunk_size):
                    replay_buffer.push(chunk)
                    samples += 1
            except Empty:
                break
        try:
            sample = replay_buffer.sample(batch_size)
            previous_state = [[item.previous_state for item in seq] for seq in sample]
            action = [[item.action for item in seq] for seq in sample]
            recipe = sample[0][0].recipe
            policy_net.train()
            q_values_selected_actions = torch.cat(
                policy_net(previous_state, action, recipe, mode="learn", hidden_states=None)
            )

            non_terminal_idxs = (~array(batch.done)).nonzero()[0]
            next_state_values = torch.zeros(len(batch.reward), device=policy_net.device)

            next_non_final_states = idx_select(batch.next_state, non_terminal_idxs)
            next_non_final_allowed_actions = idx_select(
                batch.allowed_actions, non_terminal_idxs
            )
            # recipes = idx_select(batch.recipe, non_terminal_idxs)
            policy_net.train()
            target_net.eval()
            # Double DQN here
            with torch.no_grad():
                next_state_q_values = policy_net(
                    next_non_final_states, next_non_final_allowed_actions, recipe
                )
                best_q_value_idxs = [
                    q_values.argmax().item() for q_values in next_state_q_values
                ]
                selected_actions = [
                    allowed_actions[idx]
                    for idx, allowed_actions in zip(
                        best_q_value_idxs, next_non_final_allowed_actions
                    )
                ]
                next_state_values[non_terminal_idxs] = torch.cat(
                    target_net(next_non_final_states, selected_actions, recipe)
                )
            expected_values = (
                    torch.tensor(batch.reward, device=q_values_selected_actions.device)
                    + torch.tensor(
                batch.exploration_bonus, device=q_values_selected_actions.device
            )
                    + gamma * next_state_values
            )

            optimizer.zero_grad()
            loss = smooth_l1_loss(q_values_selected_actions, expected_values)
            loss.backward()
            optimizer.step()

            t1 = q_values_selected_actions.cpu().detach().numpy().flatten()
            t2 = expected_values.cpu().detach().numpy().flatten()
            print(f"Predicted: {t1[::4]}")
            print(f"Should be: {t2[::4]}")

            if log_dir is not None:
                if learning_step % saving_freq == 0:
                    info("Saving learner weights")
                    torch.save(policy_net.state_dict(), model_path)

                writer.add_scalar("train/loss", loss.item(), learning_step)
                # writer.add_histogram(
                #     "train/learner_target_net_weights",
                #     target_net.hidden_to_scores.weight.clone().detach().cpu().numpy(),
                #     learning_step,
                # )
                # writer.add_histogram(
                #     "train/learner_policy_net_weights",
                #     policy_net.hidden_to_scores.weight.clone().detach().cpu().numpy(),
                #     learning_step,
                # )
            learning_step += 1
            info(f"Done learning step, loss={loss.item()}")

        except ValueError as e:
            print(e)
            warning("Not enough elements in buffer!")
            sleep(2.0)
