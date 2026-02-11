"""
Implémentation d'un agent DQN avec replay buffer pour MiniGrid.
"""

import gymnasium as gym
import minigrid
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from tqdm import tqdm


# Transition pour le replay buffer
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class DQN(nn.Module):
    """
    Réseau de neurones pour approximer la fonction Q.
    """
    
    def __init__(self, input_size, output_size, hidden_sizes=[128, 64]):
        """
        Args:
            input_size: Taille de l'observation
            output_size: Nombre d'actions
            hidden_sizes: Tailles des couches cachées
        """
        super(DQN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        return self.network(x)


class ReplayBuffer:
    """
    Buffer pour stocker les transitions et échantillonner des mini-batches.
    """
    
    def __init__(self, capacity=10000):
        """
        Args:
            capacity: Taille maximale du buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Ajoute une transition au buffer."""
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Échantillonne un batch aléatoire."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """Retourne la taille actuelle du buffer."""
        return len(self.buffer)


class DQNAgent:
    """
    Agent utilisant Deep Q-Network.
    
    Caractéristiques:
    - Réseau de neurones pour approximer Q
    - Replay buffer pour décorréler les expériences
    - Target network pour stabiliser l'apprentissage
    """
    
    def __init__(self,
                 input_size,
                 action_space_size,
                 hidden_sizes=[128, 64],
                 learning_rate=0.001,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 buffer_capacity=10000,
                 batch_size=64,
                 target_update_freq=10):
        """
        Initialise l'agent DQN.
        
        Args:
            input_size: Taille de l'observation
            action_space_size: Nombre d'actions
            hidden_sizes: Architecture du réseau
            learning_rate: Taux d'apprentissage
            gamma: Facteur d'actualisation
            epsilon_start, epsilon_end, epsilon_decay: Paramètres epsilon-greedy
            buffer_capacity: Taille du replay buffer
            batch_size: Taille des mini-batches
            target_update_freq: Fréquence de mise à jour du target network
        """
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation de: {self.device}")
        
        # Réseaux
        self.policy_net = DQN(input_size, action_space_size, hidden_sizes).to(self.device)
        self.target_net = DQN(input_size, action_space_size, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimiseur
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Statistiques
        self.training_rewards = []
        self.training_steps = []
        self.losses = []
        self.update_count = 0
    
    def preprocess_state(self, state):
        """
        Prétraite l'observation pour l'entrée du réseau.
        
        Args:
            state: Observation brute
            
        Returns:
            Tensor PyTorch
        """
        if isinstance(state, dict) and 'image' in state:
            state = state['image'].flatten()
        elif isinstance(state, np.ndarray):
            state = state.flatten()
        else:
            state = np.array(state).flatten()
        
        return torch.FloatTensor(state).to(self.device)
    
    def select_action(self, state, training=True):
        """
        Sélectionne une action selon epsilon-greedy.
        
        Args:
            state: État actuel
            training: Mode entraînement ou évaluation
            
        Returns:
            Action sélectionnée
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            with torch.no_grad():
                state_tensor = self.preprocess_state(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(1).item()
    
    def update(self):
        """
        Effectue une mise à jour du réseau avec un batch du replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Échantillonner un batch
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convertir en tensors
        state_batch = torch.stack([self.preprocess_state(s) for s in batch.state])
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.stack([self.preprocess_state(s) for s in batch.next_state])
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Q-values actuelles
        current_q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()
        
        # Q-values futures (avec target network)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Calculer la perte
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimisation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping pour stabilité
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.update_count += 1
        
        # Mise à jour du target network
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Décroît epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train(self, env, num_episodes=1000, max_steps=1000, verbose=True):
        """
        Entraîne l'agent DQN.
        
        Args:
            env: Environnement Gymnasium
            num_episodes: Nombre d'épisodes
            max_steps: Étapes max par épisode
            verbose: Affichage
            
        Returns:
            Historique d'entraînement
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"ENTRAÎNEMENT DQN")
            print(f"{'='*60}")
            print(f"Device: {self.device}")
            print(f"Épisodes: {num_episodes}")
            print(f"Batch size: {self.batch_size}")
            print(f"Buffer capacity: {self.replay_buffer.buffer.maxlen}")
            print(f"{'='*60}\n")
        
        episode_iterator = tqdm(range(num_episodes)) if verbose else range(num_episodes)
        
        for episode in episode_iterator:
            state, info = env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_losses = []
            
            for step in range(max_steps):
                # Sélectionner et exécuter action
                action = self.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Stocker dans le buffer
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                # Mise à jour du réseau
                loss = self.update()
                if loss is not None:
                    episode_losses.append(loss)
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
                
                if done:
                    break
            
            # Décroissance epsilon
            self.decay_epsilon()
            
            # Statistiques
            self.training_rewards.append(episode_reward)
            self.training_steps.append(episode_steps)
            if episode_losses:
                self.losses.append(np.mean(episode_losses))
            
            # Affichage périodique
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.training_rewards[-100:])
                avg_steps = np.mean(self.training_steps[-100:])
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                tqdm.write(f"Épisode {episode+1}/{num_episodes} | "
                          f"Récompense: {avg_reward:.3f} | "
                          f"Étapes: {avg_steps:.1f} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"Epsilon: {self.epsilon:.3f} | "
                          f"Buffer: {len(self.replay_buffer)}")
        
        return {
            'rewards': self.training_rewards,
            'steps': self.training_steps,
            'losses': self.losses
        }
    
    def evaluate(self, env, num_episodes=100, max_steps=1000):
        """
        Évalue l'agent.
        
        Args:
            env: Environnement
            num_episodes: Nombre d'épisodes
            max_steps: Étapes max
            
        Returns:
            Statistiques d'évaluation
        """
        rewards = []
        steps = []
        successes = 0
        
        self.policy_net.eval()
        
        for episode in range(num_episodes):
            state, info = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            for step in range(max_steps):
                action = self.select_action(state, training=False)
                state, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                
                if terminated or truncated:
                    if reward > 0:
                        successes += 1
                    break
            
            rewards.append(episode_reward)
            steps.append(episode_steps)
        
        self.policy_net.train()
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_steps': np.mean(steps),
            'std_steps': np.std(steps),
            'success_rate': successes / num_episodes,
            'rewards': rewards,
            'steps': steps
        }
    
    def save(self, filepath):
        """Sauvegarde le modèle."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_rewards': self.training_rewards,
            'training_steps': self.training_steps,
            'losses': self.losses
        }, filepath)
    
    def load(self, filepath):
        """Charge un modèle."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_rewards = checkpoint['training_rewards']
        self.training_steps = checkpoint['training_steps']
        self.losses = checkpoint['losses']


def main():
    """
    Fonction principale pour entraîner DQN.
    """
    # Environnement
    env = gym.make('MiniGrid-Empty-8x8-v0')
    
    # Obtenir la taille de l'observation
    obs, _ = env.reset()
    if isinstance(obs, dict) and 'image' in obs:
        input_size = obs['image'].flatten().shape[0]
    else:
        input_size = np.array(obs).flatten().shape[0]
    
    print(f"Taille de l'observation: {input_size}")
    
    # Créer l'agent
    agent = DQNAgent(
        input_size=input_size,
        action_space_size=env.action_space.n,
        hidden_sizes=[128, 64],
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=10
    )
    
    # Entraîner
    history = agent.train(env, num_episodes=1000, max_steps=500)
    
    # Évaluer
    print("\n" + "="*60)
    print("ÉVALUATION DQN")
    print("="*60)
    
    results = agent.evaluate(env, num_episodes=100)
    
    print(f"\nRésultats sur 100 épisodes:")
    print(f"  Récompense moyenne: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"  Étapes moyennes: {results['mean_steps']:.1f} ± {results['std_steps']:.1f}")
    print(f"  Taux de succès: {results['success_rate']*100:.1f}%")
    
    # Sauvegarder
    agent.save('dqn_agent.pth')
    print(f"\nAgent DQN sauvegardé dans 'dqn_agent.pth'")
    
    env.close()


if __name__ == "__main__":
    main()
