"""
Découverte de MiniGrid

"""

import gymnasium as gym
import minigrid
import numpy as np
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper


class MiniGridExplorer:
    """
    Classe pour explorer et comprendre l'environnement MiniGrid.
    """
    
    def __init__(self, env_name='MiniGrid-Empty-8x8-v0'):
        """
        Initialise l'environnement MiniGrid.
        
        Args:
            env_name: Nom de l'environnement à utiliser
        """
        self.env_name = env_name
        self.env = gym.make(env_name, render_mode='rgb_array')
        
        # Informations sur l'environnement
        print(f"=== Environnement: {env_name} ===")
        print(f"Espace d'observation: {self.env.observation_space}")
        print(f"Espace d'actions: {self.env.action_space}")
        print(f"Nombre d'actions: {self.env.action_space.n}")
        
    def show_action_space(self):
        """
        Affiche les actions disponibles dans MiniGrid.
        """
        actions = {
            0: "Tourner à gauche",
            1: "Tourner à droite", 
            2: "Avancer",
            3: "Ramasser un objet",
            4: "Déposer un objet",
            5: "Basculer/Activer",
            6: "Terminer"
        }
        
        print("\n=== Actions disponibles ===")
        for action_id, description in actions.items():
            if action_id < self.env.action_space.n:
                print(f"{action_id}: {description}")
    
    def explore_observation(self):
        """
        Explore la structure des observations.
        """
        obs, info = self.env.reset()
        
        print("\n=== Structure de l'observation ===")
        print(f"Type: {type(obs)}")
        print(f"Clés disponibles: {obs.keys() if isinstance(obs, dict) else 'Non-dict'}")
        
        if isinstance(obs, dict):
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"{key}: {value}")
        
        return obs, info
    
    def manual_test(self, num_steps=50):
        """
        Test manuel avec des actions aléatoires.
        
        Args:
            num_steps: Nombre de pas à effectuer
        """
        obs, info = self.env.reset()
        total_reward = 0
        
        print(f"\n=== Test manuel ({num_steps} étapes) ===")
        
        for step in range(num_steps):
            # Action aléatoire
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            total_reward += reward
            
            if reward > 0:
                print(f"Étape {step}: Récompense reçue = {reward}")
            
            if terminated or truncated:
                print(f"Épisode terminé à l'étape {step}")
                print(f"Récompense totale: {total_reward}")
                break
        
        return total_reward
    
    def analyze_rewards(self, num_episodes=10):
        """
        Analyse le système de récompenses sur plusieurs épisodes.
        
        Args:
            num_episodes: Nombre d'épisodes à tester
        """
        rewards = []
        steps = []
        
        print(f"\n=== Analyse des récompenses ({num_episodes} épisodes) ===")
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            
            for step in range(1000):  # Limite maximale
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                
                if terminated or truncated:
                    break
            
            rewards.append(episode_reward)
            steps.append(episode_steps)
            print(f"Épisode {episode+1}: Récompense={episode_reward:.2f}, Étapes={episode_steps}")
        
        print(f"\nMoyenne récompense: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Moyenne étapes: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    
    def close(self):
        """Ferme l'environnement."""
        self.env.close()


def main():
    """
    Fonction principale pour la découverte de MiniGrid.
    """
    print("="*60)
    print("TP APPRENTISSAGE PAR RENFORCEMENT - PARTIE 1")
    print("Découverte de l'environnement MiniGrid")
    print("="*60)
    
    # Créer l'explorateur
    explorer = MiniGridExplorer('MiniGrid-Empty-8x8-v0')
    
    # Afficher les actions
    explorer.show_action_space()
    
    # Explorer les observations
    explorer.explore_observation()
    
    # Test manuel
    explorer.manual_test(num_steps=50)
    
    # Analyser les récompenses
    explorer.analyze_rewards(num_episodes=10)
    
    # Fermer
    explorer.close()
    
    print("\n" + "="*60)
    print("Découverte terminée avec succès!")
    print("="*60)


if __name__ == "__main__":
    main()
