"""
An extension of https://github.com/qzed/irl-maxent/blob/master/src/maxent.py for
soft constrained irl.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score


class ConstraintNetwork(nn.Module):
    def __init__(self, feature_dim):
        super(ConstraintNetwork, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 128)  # Input is now the feature vector
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Single output: probability of being constrained

    def forward(self, features):
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))  # Output is a probability

def pretrain_constraint_network(constraint_net, trajectories, feature_extractor, epochs=50, lr=0.001):
    """Pretrain the constraint network on expert trajectories using extracted feature vectors."""
    optimizer = optim.Adam(constraint_net.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Prepare training data from expert trajectories using the feature extractor
    data = []
    labels = []
    for t in trajectories:
        for s, a, s_ in t.transitions():
            feature_vector = feature_extractor(s, a, s_)  # Extract the feature vector for the (s, a, s') transition
            data.append(torch.tensor(feature_vector, dtype=torch.float32))
            labels.append(0)  # Label as unconstrained (expert follows constraints)

    data = torch.stack(data)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Shape: (N, 1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = constraint_net(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # Convert network outputs and labels to binary predictions
        preds = (output >= 0.5).float()  # Threshold at 0.5 for binary classification
        true_labels = labels

        # Calculate precision and recall
        precision = precision_score(true_labels.detach().numpy(), preds.detach().numpy())
        recall = recall_score(true_labels.detach().numpy(), preds.detach().numpy())

        if epoch % 2 == 0:
            print(f"Pretraining Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")


def backward_causal(p_transition, reward, terminal, discount, eps=1e-5):
    n_states, _, n_actions = p_transition.shape

    # set up terminal reward function
    if len(terminal) == n_states:
        reward_terminal = np.array(terminal, dtype=np.float)
    else:
        reward_terminal = -np.inf * np.ones(n_states)
        reward_terminal[terminal] = 0.0

    # compute state log partition V and state-action log partition Q
    v = -1e200 * np.ones(n_states)  # np.dot doesn't behave with -np.inf

    p_t = discount * np.moveaxis(p_transition, 1, 2)
    r = (reward * p_t).sum(-1)  # computes the state-action rewards

    delta = np.inf

    while delta > eps:
        v_old = v
        q = r + p_t @ v_old

        v = reward_terminal
        for a in range(n_actions):
            v = _softmax(v, q[:, a])

        delta = np.max(np.abs(v - v_old))

    # compute and return policy
    return np.exp(q - v[:, None])


def forward(p_transition, p_initial, policy, terminal, eps=1e-5):
    # Don't allow transitions from terminal
    p_t = np.moveaxis(p_transition.copy(), 1, 2)
    p_terminal = p_t[terminal, :, :].copy()
    p_t[terminal, :, :] = 0.0

    d = p_initial.sum(1)
    d_total = d

    delta = np.inf
    while delta > eps:
        # state-action expected visitation
        d_sa = d[:, None] * policy
        # for each state s, multiply the expected visitations of all states to s by their probabilities
        # d_s = sum(sa_ev[s_from, a] * p_t[s_from, a, s] for all s_from, a)
        d_ = (d_sa[:, :, None] * p_t).sum((0, 1))

        delta = np.max(np.abs(d - d_))
        d = d_
        d_total += d

    p_t[terminal, :, :] = p_terminal
    # Distribute the visitation stats of satate to their actions
    d_sa = d_total[:, None] * policy

    # Distribute state-action visitations to the next states
    d_transition = d_sa[:, :, None] * p_t

    return d_transition


def initial_probabilities(n_states, n_actions, trajectories):
    initial = np.zeros((n_states, n_actions))
    for t in trajectories:
        s, a, _ = list(t.transitions())[0]
        initial[s, a] += 1
    return initial / len(trajectories)


def ef_from_trajectories(features, trajectories):
    n_features = features.shape[-1]

    fe = np.zeros(n_features)

    for t in trajectories:
        for s, a, s_ in t.transitions():
            fe += features[s, a, s_, :]

    return fe / len(trajectories)


def dicrl(nominal_rewards, p_transition, features, terminal, trajectories, optim, init, discount,
         eps=1e-4, eps_error=1e-2, burnout=100, max_iter=10000, max_penalty=200, log=None, initial_omega=None):

    print('Starting DICRL...')
    n_states, n_actions, _, n_features = features.shape
    feature_dim = n_features

    constraint_net = ConstraintNetwork(feature_dim=feature_dim)

    def feature_extractor(s, a, s_):
        return features[s, a, s_, :]
    #pretrain constrain nn with expert trajectories
    pretrain_constraint_network(constraint_net, trajectories, feature_extractor)
    constraint_net.eval()

    # Don't count transitions that start with a terminal state
    features[terminal] = 0

    # compute static properties from trajectories
    e_features = ef_from_trajectories(features, trajectories)
    p_initial = initial_probabilities(n_states, n_actions, trajectories)
    nominal_rewards = np.array(nominal_rewards)

    omega = init(n_features)
    if initial_omega is not None:
        omega = initial_omega.copy()
    delta = mean_error = np.inf

    optim.reset(omega)
    epoch = 0
    best = None
    best_error = 100000
    while epoch <= burnout or (delta > eps and mean_error > eps_error and epoch < max_iter):
        omega_old = omega.copy()

        # compute per-state reward
        reward = nominal_rewards.copy()
        for s in range(n_states):
            for a in range(n_actions):
                for s_ in range(n_states):
                    # Extract feature vector for (s, a, s') transition
                    feature_vector = torch.tensor(feature_extractor(s, a, s_), dtype=torch.float32)
                    with torch.no_grad():
                        constraint_prob = constraint_net(feature_vector).item()
                        reward[s, a] -= constraint_prob * 50  # Adjust penalty as needed
        # reward = constraint_net(s, a, s_)
        # REPLACE ABOVE with nn forward pass

        # Backward, Forward
        # REPLACE BACKWARD STEP WITH NN ?
        policy = backward_causal(p_transition, reward, terminal, discount)
        d = forward(p_transition, p_initial, policy, terminal)

        # compute the gradient
        # df[i] is the expected visitation for feature i accross all (s, a, s_)
        # df[i] = [d[s, a, s_, i] * features[s, a, s_, i] for all (s, a, s_)].sum()
        df = (d[:, :, :, None] * features).sum((0, 1, 2))
        grad = df - e_features

        mean_error = np.abs(grad).mean()

        if epoch >= burnout and mean_error < best_error:
            best = omega.copy()
            best_error = mean_error

        # perform optimization step and compute delta for convergence
        optim.step(grad)

        if omega.max() > max_penalty:
            omega = omega * (max_penalty / omega.max())
            optim.reset(omega)

        delta = np.max(np.abs(omega_old - omega))

        if log is not None and type(log) == list:
            log.append({
                'omega': omega_old.copy(),
                'delta': delta,
                'epoch': epoch,
                'mean_error': mean_error,
                'is_best': mean_error == best_error,
                'best_omega': omega_old.copy() if best is None else best.copy(),
                'best_reward': reward
            })

        if epoch % 100 == 0:
            print(f'MAE(best): {min(mean_error, best_error): 0.15f}')
        epoch += 1

    print(f'Finished with MAE(best): {best_error: 0.15f}')

    return best if best is not None else omega


def _softmax(x1, x2):
    x_max = np.maximum(x1, x2)
    x_min = np.minimum(x1, x2)
    return x_max + np.log(1.0 + np.exp(x_min - x_max))
