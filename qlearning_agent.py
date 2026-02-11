"""
TP Apprentissage par Renforcement - Partie 2: Implémentation Q-Learning
Auteur: [Votre Nom]
Date: Février 2026

Implémentation d'un agent Q-Learning pour MiniGrid.
"""

import gymnasium as gym
import minigrid
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm


class QLearningAgent:
    """
    Agent utilisant l'algorithme Q-Learning tabulaire.
    
    Caractéristiques:
    - Table Q pour stocker les valeurs état-action
    - Exploration epsilon-greedy
    - Mise à jour selon la règle de Bellman
    """
    
    def __init__(self, 
                 action_space_size,
                 learning_rate=0.1,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995):
        """
        Initialise l'agent Q-Learning.
        
        Args:
            action_space_size: Nombre d'actions possibles
            learning_rate: Taux d'apprentissage (alpha)
            gamma: Facteur d'actualisation
            epsilon_start: Valeur initiale d'epsilon
            epsilon_end: Valeur finale d'epsilon
            epsilon_decay: Taux de décroissance d'epsilon
        """
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Table Q: dictionnaire état -> array de Q-values
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))
        
        # Statistiques
        self.training_rewards = []
        self.training_steps = []
        
    def state_to_key(self, observation):
        """
        Convertit une observation en clé hashable pour la table Q.
        
        Pour MiniGrid, on utilise la position de l'agent et sa direction.
        
        Args:
            observation: Observation de l'environnement
            
        Returns:
            Tuple représentant l'état
        """
        if isinstance(observation, dict):
            # Extraire la position de l'agent depuis l'image d'observation
            # Simplification: on utilise l'image entière comme état
            if 'image' in observation:
                return tuple(observation['image'].flatten())
            elif 'agent_pos' in observation:
                return tuple(observation['agent_pos'])
        
        # Si observation est déjà un array
        if isinstance(observation, np.ndarray):
            return tuple(observation.flatten())
        
        return tuple(observation)
    
    def select_action(self, state, training=True):
        """
        Sélectionne une action selon la politique epsilon-greedy.
        
        Args:
            state: État actuel
            training: Si True, utilise epsilon-greedy, sinon greedy pur
            
        Returns:
            Action sélectionnée
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: action aléatoire
            return np.random.randint(self.action_space_size)
        else:
            # Exploitation: meilleure action
            state_key = self.state_to_key(state)
            return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state, done):
        """
        Met à jour la table Q selon la règle de Q-Learning.
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: État actuel
            action: Action effectuée
            reward: Récompense reçue
            next_state: État suivant
            done: Episode terminé ou non
        """
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        
        # Valeur actuelle
        current_q = self.q_table[state_key][action]
        
        # Meilleure valeur future
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_state_key])
        
        # Mise à jour Q-Learning
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self.learning_rate * td_error
        
        self.q_table[state_key][action] = new_q
    
    def decay_epsilon(self):
        """Décroît epsilon pour réduire l'exploration au fil du temps."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train(self, env, num_episodes=1000, max_steps=1000, verbose=True):
        """
        Entraîne l'agent sur l'environnement.
        
        Args:
            env: Environnement Gymnasium
            num_episodes: Nombre d'épisodes d'entraînement
            max_steps: Nombre maximum d'étapes par épisode
            verbose: Afficher la progression
            
        Returns:
            Historique de l'entraînement
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"ENTRAÎNEMENT Q-LEARNING")
            print(f"{'='*60}")
            print(f"Épisodes: {num_episodes}")
            print(f"Alpha (learning rate): {self.learning_rate}")
            print(f"Gamma (discount): {self.gamma}")
            print(f"Epsilon: {self.epsilon} → {self.epsilon_end}")
            print(f"{'='*60}\n")
        
        episode_iterator = tqdm(range(num_episodes)) if verbose else range(num_episodes)
        
        for episode in episode_iterator:
            state, info = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            for step in range(max_steps):
                # Sélectionner et exécuter une action
                action = self.select_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Mise à jour Q-Learning
                self.update(state, action, reward, next_state, done)
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
                
                if done:
                    break
            
            # Décroissance epsilon
            self.decay_epsilon()
            
            # Enregistrer les statistiques
            self.training_rewards.append(episode_reward)
            self.training_steps.append(episode_steps)
            
            # Affichage périodique
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.training_rewards[-100:])
                avg_steps = np.mean(self.training_steps[-100:])
                tqdm.write(f"Épisode {episode+1}/{num_episodes} | "
                          f"Récompense moy: {avg_reward:.3f} | "
                          f"Étapes moy: {avg_steps:.1f} | "
                          f"Epsilon: {self.epsilon:.3f}")
        
        return {
            'rewards': self.training_rewards,
            'steps': self.training_steps
        }
    
    def evaluate(self, env, num_episodes=100, max_steps=1000):
        """
        Évalue l'agent entraîné.
        
        Args:
            env: Environnement Gymnasium
            num_episodes: Nombre d'épisodes d'évaluation
            max_steps: Nombre maximum d'étapes par épisode
            
        Returns:
            Dictionnaire avec les statistiques d'évaluation
        """
        rewards = []
        steps = []
        successes = 0
        
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
                    if reward > 0:  # Succès si récompense positive
                        successes += 1
                    break
            
            rewards.append(episode_reward)
            steps.append(episode_steps)
        
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
        """Sauvegarde l'agent."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
                'training_rewards': self.training_rewards,
                'training_steps': self.training_steps
            }, f)
    
    def load(self, filepath):
        """Charge un agent sauvegardé."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.action_space_size), data['q_table'])
            self.epsilon = data['epsilon']
            self.training_rewards = data['training_rewards']
            self.training_steps = data['training_steps']


def main():
    """
    Fonction principale pour entraîner l'agent Q-Learning.
    """
    # Créer l'environnement
    env = gym.make('MiniGrid-Empty-8x8-v0')
    
    # Créer l'agent
    agent = QLearningAgent(
        action_space_size=env.action_space.n,
        learning_rate=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    
    # Entraîner
    history = agent.train(env, num_episodes=1000, max_steps=500)
    
    # Évaluer
    print("\n" + "="*60)
    print("ÉVALUATION")
    print("="*60)
    
    results = agent.evaluate(env, num_episodes=100)
    
    print(f"\nRésultats sur 100 épisodes:")
    print(f"  Récompense moyenne: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"  Étapes moyennes: {results['mean_steps']:.1f} ± {results['std_steps']:.1f}")
    print(f"  Taux de succès: {results['success_rate']*100:.1f}%")
    
    # Sauvegarder
    agent.save('qlearning_agent.pkl')
    print(f"\nAgent sauvegardé dans 'qlearning_agent.pkl'")
    
    env.close()


if __name__ == "__main__":
    main()
