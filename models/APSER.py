import torch
import numpy as np

class SumTree(object):
    def __init__(self, max_size):
        self.levels = [np.zeros(1)]
        # Tree construction
        # Double the number of nodes at each level
        level_size = 1
        while level_size < max_size:
            level_size *= 2
            self.levels.append(np.zeros(level_size))

    # Batch binary search through sum tree
    # Sample a priority between 0 and the max priority and then search the tree for the corresponding index
    def sample(self, batch_size):
        value = np.random.uniform(0, self.levels[0][0], size=batch_size)
        ind = np.zeros(batch_size, dtype=int)

        for nodes in self.levels[1:]:
            ind *= 2
            left_sum = nodes[ind]

            is_greater = np.greater(value, left_sum)

            # If value > left_sum -> go right (+1), else go left (+0)
            ind += is_greater

            # If we go right, we only need to consider the values in the right tree
            # so we subtract the sum of values in the left tree
            value -= left_sum * is_greater

        return ind

    def set(self, ind, new_priority):
        priority_diff = new_priority - self.levels[-1][ind]

        for nodes in self.levels[::-1]:
            np.add.at(nodes, ind, priority_diff)
            ind //= 2

    def batch_set(self, ind, new_priority):
        # Confirm we don't increment a node twice
        ind, unique_ind = np.unique(ind, return_index=True)
        priority_diff = new_priority[unique_ind] - self.levels[-1][ind]

        for nodes in self.levels[::-1]:
            np.add.at(nodes, ind, priority_diff)
            ind //= 2


class PrioritizedReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device=None):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.tree = SumTree(max_size)
        self.max_priority = 1.0
        self.beta = 0.4

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.tree.set(self.ptr, self.max_priority)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = self.tree.sample(batch_size)

        weights = self.tree.levels[-1][ind] ** -self.beta
        weights /= weights.max()

        self.beta = min(self.beta + 2e-7, 1)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            ind,
            torch.FloatTensor(weights).to(self.device).reshape(-1, 1)
        )

    def update_priority(self, ind, priority):
        self.max_priority = max(priority.max(), self.max_priority)
        self.tree.batch_set(ind, priority)



def APSER(replay_buffer: PrioritizedReplayBuffer, agent, batch_size, beta, discount, ro, max_steps_before_truncation: int, bootstrap_steps=1, env=None, update_neigbors= False, uniform_sampling=False, normalize = True, sigmoid=False):
    states, actions, next_states, rewards, not_dones, indices, weights = replay_buffer.sample(batch_size)
    predicted_actions = torch.FloatTensor(agent.select_action(states)).to(agent.device).reshape(states.shape[0], -1)
    next_actions = torch.FloatTensor(np.array([replay_buffer.action[indices[i]+1] for i in range(batch_size)])).to(agent.device)
    previous_scores_with_current_critic = rewards + discount * not_dones * agent.critic.Q1(next_states, next_actions).detach()
    current_scores_with_current_critic = agent.critic.Q1(states, predicted_actions).detach()
    # Calculate improvement and priority for batch
    improvements = (current_scores_with_current_critic - previous_scores_with_current_critic)
    if normalize: 
        improvements = (improvements - improvements.mean()) / (improvements.std() + 1e-5)

    if sigmoid:
        priorities = torch.sigmoid(improvements).T.detach().cpu().numpy()[0]
    else:
        priorities = improvements.T.detach().cpu().numpy()[0]
    # Update priorities in replay buffer in batch
    replay_buffer.update_priority(indices, priorities)
    if update_neigbors:
        root = 1
        nb_neighbors_to_update = (np.abs(priorities) * max_steps_before_truncation ** root).astype(int)
        # Create a vector of all neighbor indices
        for i, nb_neighbors in enumerate(nb_neighbors_to_update):
            if nb_neighbors < 2:
                continue
            neighbors_before = np.clip(indices[i] - np.arange(1, nb_neighbors // 2 + 1), 0, replay_buffer.size - 1)
            neighbors_after = np.clip(indices[i] + np.arange(1, nb_neighbors // 2 + 1), 0, replay_buffer.size - 1)
            before_indexes = (replay_buffer.not_done[neighbors_before] == 0)
            if any(before_indexes):
                neighbors_before = neighbors_before[:list(before_indexes[::-1]).index(True)]
            after_indexes = (replay_buffer.not_done[neighbors_after] == 0)
            if any(after_indexes):
                neighbors_after = neighbors_after[:list(after_indexes).index(True)]
            # Concatenate neighbors and compute priorities in a vectorized manner
            all_neighbors = np.concatenate([neighbors_before, neighbors_after])
            normalized_priority = priorities[i]
            all_priorities = np.sign(normalized_priority) * np.concatenate([np.abs(normalized_priority) * ro ** np.arange(1, len(neighbors_before)+1), np.abs(normalized_priority) * ro ** np.arange(1, len(neighbors_after)+1)])
            # Vectorized update of neighbors
            current_priorities = replay_buffer.tree.levels[-1][all_neighbors]
            updated_priorities = np.clip(current_priorities + all_priorities, 0, replay_buffer.max_priority)
            if all_neighbors.size > 0:
                replay_buffer.update_priority(all_neighbors, updated_priorities)
    return states, actions, next_states, rewards, not_dones, weights, indices

class ExperienceReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device=None):
        self.device = device

        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )