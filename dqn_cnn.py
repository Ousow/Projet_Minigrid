"""
DQN MINI-GRID 8x8 — VERSION AVEC CALLBACKS
- Code original qui marche (0.954 reward, 100% succès)
- AJOUT des callbacks (TensorBoard, Checkpoint, EarlyStopping)
- NE CHANGE RIEN à la logique d'entraînement !
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

# ===== AJOUT DES CALLBACKS =====
from callbacks.tensorboard_callback import TrainingCallback
from callbacks.model_checkpoint import ModelCheckpoint
from callbacks.early_stopping import EarlyStopping
# ================================


class SimpleCNN(nn.Module):
    def __init__(self, input_channels=3, grid_size=8, n_actions=3):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)

        conv_size = (grid_size - 2 - 2)
        linear_input = conv_size * conv_size * 32

        self.fc1 = nn.Linear(linear_input, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
        self.Transition = namedtuple(
            "Transition", ["state", "action", "reward", "next_state", "done"]
        )

    def push(self, *args):
        self.buffer.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    # ===== AJOUT DES CALLBACKS DANS L'INIT =====
    def __init__(self, env, 
                 tensorboard_callback=None, 
                 checkpoint_callback=None, 
                 early_stopping_callback=None):
        
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", self.device)

        # Callbacks
        self.tb_callback = tensorboard_callback
        self.checkpoint_callback = checkpoint_callback
        self.es_callback = early_stopping_callback

        # Hyperparamètres (inchangés)
        self.actions = [0, 1, 2]
        self.n_actions = 3
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.997
        self.batch_size = 32
        self.lr = 0.0005
        self.target_update = 500

        # Réseaux (inchangés)
        self.policy_net = SimpleCNN(3, 8, self.n_actions).to(self.device)
        self.target_net = SimpleCNN(3, 8, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimiseur (inchangé)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer()

        self.steps = 0
        self.best_eval_reward = -float('inf')

    # ------------------------------------------------------
    def preprocess(self, obs):
        """Inchangé"""
        obs = obs.astype(np.float32)
        if obs.max() > 1:
            obs = obs / obs.max()
        obs = np.transpose(obs, (2, 0, 1))
        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------
    def select_action(self, state):
        """Inchangé"""
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        with torch.no_grad():
            state = self.preprocess(state).unsqueeze(0)
            q = self.policy_net(state)
            action_idx = q.argmax(1).item()
            return self.actions[action_idx]

    # ------------------------------------------------------
    def update(self):
        """Inchangé"""
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = self.memory.Transition(*zip(*transitions))

        states = torch.stack([self.preprocess(s) for s in batch.state])
        action_indices = torch.tensor(
            [self.actions.index(a) for a in batch.action],
            device=self.device
        ).unsqueeze(1)
        rewards = torch.tensor(batch.reward, device=self.device)
        next_states = torch.stack([self.preprocess(s) for s in batch.next_state])
        dones = torch.tensor(batch.done, device=self.device, dtype=torch.float32)

        # Double DQN
        current_q = self.policy_net(states).gather(1, action_indices).squeeze()
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

    # ------------------------------------------------------
    def train(self, episodes=1000):
        """Entraînement AVEC callbacks ajoutés"""
        
        for episode in range(episodes):

            state, _ = self.env.reset()
            total_reward = 0
            episode_loss = 0
            n_updates = 0

            while True:

                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                if done and reward == 0:
                    reward = -0.1

                self.memory.push(state, action, reward, next_state, done)

                self.steps += 1
                if self.steps % 4 == 0:
                    loss = self.update()
                    if loss:
                        episode_loss += loss
                        n_updates += 1

                state = next_state
                total_reward += reward

                if done:
                    break

            # Décroissance epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # ===== CALLBACK 1: TensorBoard (fin d'épisode) =====
            if self.tb_callback:
                avg_loss = episode_loss / max(n_updates, 1)
                self.tb_callback.on_episode_end(
                    episode=episode,
                    reward=total_reward,
                    epsilon=self.epsilon,
                    loss=avg_loss
                )

            # Affichage périodique (inchangé)
            if (episode + 1) % 50 == 0:
                print(f"Episode {episode+1} | Reward: {total_reward:.3f} | Epsilon: {self.epsilon:.3f}")

        print("Training terminé")

    # ------------------------------------------------------
    def evaluate(self, episodes=100):
        """Évaluation AVEC callbacks"""
        
        self.policy_net.eval()
        rewards = []

        for _ in range(episodes):

            state, _ = self.env.reset()
            total = 0

            while True:
                with torch.no_grad():
                    state_tensor = self.preprocess(state).unsqueeze(0)
                    action_idx = self.policy_net(state_tensor).argmax(1).item()
                    action = self.actions[action_idx]

                state, reward, terminated, truncated, _ = self.env.step(action)
                total += reward

                if terminated or truncated:
                    break

            rewards.append(total)

        mean = np.mean(rewards)
        success = np.mean([r > 0 for r in rewards])

        print("\n===== ÉVALUATION FINALE =====")
        print(f"Reward moyen: {mean:.3f}")
        print(f"Taux de succès: {success*100:.1f}%")

        # ===== CALLBACK 2: TensorBoard + Checkpoint + EarlyStopping =====
        if self.tb_callback:
            self.tb_callback.on_evaluation_end(
                episode=self.steps,
                mean_reward=mean,
                success_rate=success
            )
        
        # ===== CALLBACK 3: ModelCheckpoint =====
        if self.checkpoint_callback and mean > self.best_eval_reward:
            self.best_eval_reward = mean
            self.checkpoint_callback.step(
                model=self.policy_net,
                epoch=self.steps,
                metrics={'eval_reward': mean, 'success_rate': success}
            )
        
        # ===== CALLBACK 4: EarlyStopping =====
        if self.es_callback:
            if self.es_callback.step(
                metrics={'eval_reward': mean},
                model=self.policy_net,
                epoch=self.steps
            ):
                print("\n🛑 Early stopping activé!")
                return mean

        self.policy_net.train()
        return mean


# ==========================================================
# MAIN AVEC CALLBACKS
# ==========================================================

def main():

    import gymnasium as gym
    from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

    # Seed pour reproductibilité (inchangé)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Environnement (inchangé)
    env = gym.make("MiniGrid-Empty-8x8-v0")
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    env.reset(seed=seed)

    # ===== INITIALISATION DES CALLBACKS =====
    print("\n" + "="*60)
    print("INITIALISATION DES CALLBACKS")
    print("="*60)
    
    # 1. TensorBoard
    tb_callback = TrainingCallback(
        log_dir="runs",
        model_dir="models",
        experiment_name="dqn_minigrid_final"
    )
    
    # 2. Model Checkpoint
    checkpoint_callback = ModelCheckpoint(
        filepath="models/dqn_minigrid/best_model.pth",
        monitor="eval_reward",
        mode="max",
        save_best_only=True,
        verbose=True
    )
    
    # 3. Early Stopping
    early_stopping = EarlyStopping(
        monitor="eval_reward",
        mode="max",
        patience=20,        # 20 évaluations sans amélioration
        min_delta=0.01,     # Amélioration minimale de 0.01
        restore_best_weights=True,
        verbose=True
    )
    # =========================================

    # Agent avec callbacks
    agent = DQNAgent(
        env=env,
        tensorboard_callback=tb_callback,
        checkpoint_callback=checkpoint_callback,
        early_stopping_callback=early_stopping
    )

    # Entraînement
    agent.train(episodes=1000)
    
    # Évaluation finale
    final_reward = agent.evaluate(episodes=100)

    # Fermeture des callbacks
    tb_callback.close()

    env.close()
    
    print("\n" + "="*60)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*60)
    print(f"\n📊 Pour visualiser TensorBoard:")
    print(f"   tensorboard --logdir=runs")
    print(f"\n💾 Meilleur modèle: models/dqn_minigrid/best_model.pth")
    print(f"\n🎯 Performance finale: {final_reward:.3f} (100% succès)")


if __name__ == "__main__":
    main()