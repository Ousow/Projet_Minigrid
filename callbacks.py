"""
Système de Callbacks pour DQN flexible pour surveiller
et contrôler l'entraînement de l'agent DQN.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json
from datetime import datetime


class Callback:
    """Classe de base pour tous les callbacks."""
    
    def on_training_begin(self, agent, env):
        """Appelé au début de l'entraînement."""
        pass
    
    def on_training_end(self, agent, env):
        """Appelé à la fin de l'entraînement."""
        pass
    
    def on_episode_begin(self, episode, agent, env):
        """Appelé au début de chaque épisode."""
        pass
    
    def on_episode_end(self, episode, agent, env, episode_info):
        """Appelé à la fin de chaque épisode."""
        pass
    
    def on_step(self, episode, step, agent, state, action, reward, next_state, done):
        """Appelé à chaque step."""
        pass


class EarlyStopping(Callback):
    """
    Arrête l'entraînement si la performance ne s'améliore plus.
    """
    
    def __init__(self, patience=50, min_delta=0.01, metric='reward'):
        """
        Args:
            patience: Nombre d'épisodes à attendre sans amélioration
            min_delta: Amélioration minimale pour considérer qu'il y a progrès
            metric: Métrique à surveiller ('reward' ou 'steps')
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.best_value = -np.inf if metric == 'reward' else np.inf
        self.wait = 0
        self.stopped_episode = None
    
    def on_episode_end(self, episode, agent, env, episode_info):
        """Vérifie si on doit arrêter."""
        current_value = episode_info[self.metric]
        
        # Calculer si c'est une amélioration
        if self.metric == 'reward':
            improved = current_value > self.best_value + self.min_delta
        else:  # steps (on veut minimiser)
            improved = current_value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
        
        # Arrêter si patience dépassée
        if self.wait >= self.patience:
            self.stopped_episode = episode
            print(f"\n Early stopping déclenché à l'épisode {episode}")
            print(f"   Pas d'amélioration depuis {self.patience} épisodes")
            print(f"   Meilleure {self.metric}: {self.best_value:.3f}")
            return True  # Signal pour arrêter
        
        return False


class ModelCheckpoint(Callback):
    """
    Sauvegarde le meilleur modèle pendant l'entraînement.
    """
    
    def __init__(self, filepath='best_model.pth', monitor='reward', 
                 save_best_only=True, verbose=True):
        """
        Args:
            filepath: Chemin pour sauvegarder le modèle
            monitor: Métrique à surveiller ('reward' ou 'steps')
            save_best_only: Sauvegarder uniquement le meilleur modèle
            verbose: Afficher les messages
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best_value = -np.inf if monitor == 'reward' else np.inf
        self.episodes_since_improvement = 0
    
    def on_episode_end(self, episode, agent, env, episode_info):
        """Sauvegarde si c'est le meilleur modèle."""
        # Calculer moyenne sur les derniers 100 épisodes
        if len(agent.training_rewards) >= 100:
            current_value = np.mean(agent.training_rewards[-100:])
        else:
            current_value = np.mean(agent.training_rewards)
        
        # Vérifier si c'est une amélioration
        if self.monitor == 'reward':
            is_improvement = current_value > self.best_value
        else:  # steps
            if len(agent.training_steps) >= 100:
                current_value = np.mean(agent.training_steps[-100:])
            else:
                current_value = np.mean(agent.training_steps)
            is_improvement = current_value < self.best_value
        
        if is_improvement or not self.save_best_only:
            if is_improvement:
                self.best_value = current_value
                self.episodes_since_improvement = 0
            else:
                self.episodes_since_improvement += 1
            
            # Sauvegarder
            agent.save(self.filepath)
            
            if self.verbose and is_improvement:
                print(f"\n Meilleur modèle sauvegardé (ép. {episode})")
                print(f"   {self.monitor}: {current_value:.3f}")


class ProgressTracker(Callback):
    """
    Affiche la progression pendant l'entraînement.
    """
    
    def __init__(self, print_freq=100):
        """
        Args:
            print_freq: Fréquence d'affichage (en épisodes)
        """
        self.print_freq = print_freq
        self.start_time = None
    
    def on_training_begin(self, agent, env):
        """Initialise le timer."""
        self.start_time = datetime.now()
        print(f"\n{'='*70}")
        print(f"DÉBUT DE L'ENTRAÎNEMENT - {self.start_time.strftime('%H:%M:%S')}")
        print(f"{'='*70}\n")
    
    def on_episode_end(self, episode, agent, env, episode_info):
        """Affiche la progression."""
        if (episode + 1) % self.print_freq == 0:
            avg_reward = np.mean(agent.training_rewards[-100:])
            avg_steps = np.mean(agent.training_steps[-100:])
            
            elapsed = datetime.now() - self.start_time
            
            print(f"\n Épisode {episode+1}")
            print(f"   Récompense (100 derniers): {avg_reward:.3f}")
            print(f"   Étapes (100 derniers): {avg_steps:.1f}")
            print(f"   Epsilon: {agent.epsilon:.3f}")
            print(f"   Buffer: {len(agent.replay_buffer)}/{agent.replay_buffer.buffer.maxlen}")
            print(f"   Temps écoulé: {elapsed}")


class LivePlotter(Callback):
    """
    Affiche des graphiques en temps réel pendant l'entraînement.
    """
    
    def __init__(self, plot_freq=200, window=100):
        """
        Args:
            plot_freq: Fréquence de mise à jour (en épisodes)
            window: Taille de la fenêtre pour la moyenne mobile
        """
        self.plot_freq = plot_freq
        self.window = window
        self.fig = None
        self.axes = None
    
    def on_training_begin(self, agent, env):
        """Initialise la figure."""
        plt.ion()  # Mode interactif
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('Entraînement DQN - Live', fontsize=16, fontweight='bold')
    
    def on_episode_end(self, episode, agent, env, episode_info):
        """Met à jour les graphiques."""
        if (episode + 1) % self.plot_freq == 0 and len(agent.training_rewards) > self.window:
            self._update_plots(agent)
    
    def _update_plots(self, agent):
        """Met à jour tous les graphiques."""
        for ax in self.axes.flat:
            ax.clear()
        
        episodes = range(1, len(agent.training_rewards) + 1)
        
        # Récompenses
        ax = self.axes[0, 0]
        ax.plot(episodes, agent.training_rewards, alpha=0.3, color='blue')
        if len(agent.training_rewards) >= self.window:
            moving_avg = np.convolve(agent.training_rewards, 
                                    np.ones(self.window)/self.window, mode='valid')
            ax.plot(range(self.window, len(agent.training_rewards)+1), moving_avg,
                   color='red', linewidth=2)
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Récompense')
        ax.set_title('Récompenses')
        ax.grid(True, alpha=0.3)
        
        # Étapes
        ax = self.axes[0, 1]
        ax.plot(episodes, agent.training_steps, alpha=0.3, color='green')
        if len(agent.training_steps) >= self.window:
            moving_avg = np.convolve(agent.training_steps,
                                    np.ones(self.window)/self.window, mode='valid')
            ax.plot(range(self.window, len(agent.training_steps)+1), moving_avg,
                   color='orange', linewidth=2)
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Étapes')
        ax.set_title('Nombre d\'étapes par épisode')
        ax.grid(True, alpha=0.3)
        
        # Loss
        ax = self.axes[1, 0]
        if agent.losses:
            ax.plot(agent.losses, alpha=0.3, color='purple')
            if len(agent.losses) >= 50:
                moving_avg = np.convolve(agent.losses, np.ones(50)/50, mode='valid')
                ax.plot(range(50, len(agent.losses)+1), moving_avg,
                       color='darkviolet', linewidth=2)
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Loss')
        ax.set_title('Loss du réseau')
        ax.grid(True, alpha=0.3)
        
        # Epsilon
        ax = self.axes[1, 1]
        epsilon_history = [agent.epsilon_start * (agent.epsilon_decay ** i) 
                          for i in range(len(agent.training_rewards))]
        epsilon_history = [max(e, agent.epsilon_end) for e in epsilon_history]
        ax.plot(episodes, epsilon_history, color='darkgreen', linewidth=2)
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Epsilon')
        ax.set_title('Taux d\'exploration (Epsilon)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.001)
    
    def on_training_end(self, agent, env):
        """Sauvegarde la figure finale."""
        if self.fig:
            plt.ioff()
            self.fig.savefig('training_progress.png', dpi=300, bbox_inches='tight')
            print("\n Graphiques sauvegardés: training_progress.png")


class MetricsLogger(Callback):
    """
    Enregistre toutes les métriques dans un fichier JSON.
    """
    
    def __init__(self, filepath='training_metrics.json', save_freq=100):
        """
        Args:
            filepath: Chemin du fichier JSON
            save_freq: Fréquence de sauvegarde (en épisodes)
        """
        self.filepath = filepath
        self.save_freq = save_freq
        self.metrics = {
            'episodes': [],
            'rewards': [],
            'steps': [],
            'losses': [],
            'epsilon': [],
            'success': []
        }
    
    def on_episode_end(self, episode, agent, env, episode_info):
        """Enregistre les métriques."""
        self.metrics['episodes'].append(episode)
        self.metrics['rewards'].append(episode_info['reward'])
        self.metrics['steps'].append(episode_info['steps'])
        if agent.losses:
            self.metrics['losses'].append(agent.losses[-1])
        self.metrics['epsilon'].append(agent.epsilon)
        self.metrics['success'].append(episode_info.get('success', False))
        
        # Sauvegarder périodiquement
        if (episode + 1) % self.save_freq == 0:
            self._save()
    
    def _save(self):
        """Sauvegarde les métriques."""
        with open(self.filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def on_training_end(self, agent, env):
        """Sauvegarde finale."""
        self._save()
        print(f"\n Métriques sauvegardées: {self.filepath}")


class LearningRateScheduler(Callback):
    """
    Ajuste le learning rate pendant l'entraînement.
    """
    
    def __init__(self, schedule='step', factor=0.5, patience=200):
        """
        Args:
            schedule: Type de schedule ('step', 'exponential', 'plateau')
            factor: Facteur de réduction
            patience: Patience pour 'plateau'
        """
        self.schedule = schedule
        self.factor = factor
        self.patience = patience
        self.wait = 0
        self.best_reward = -np.inf
        self.initial_lr = None
    
    def on_training_begin(self, agent, env):
        """Enregistre le LR initial."""
        self.initial_lr = agent.optimizer.param_groups[0]['lr']
    
    def on_episode_end(self, episode, agent, env, episode_info):
        """Ajuste le learning rate."""
        if self.schedule == 'step' and (episode + 1) % 500 == 0:
            self._reduce_lr(agent)
        
        elif self.schedule == 'plateau':
            avg_reward = np.mean(agent.training_rewards[-100:])
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self._reduce_lr(agent)
                    self.wait = 0
    
    def _reduce_lr(self, agent):
        """Réduit le learning rate."""
        current_lr = agent.optimizer.param_groups[0]['lr']
        new_lr = current_lr * self.factor
        agent.optimizer.param_groups[0]['lr'] = new_lr
        print(f"\n Learning rate réduit: {current_lr:.6f} → {new_lr:.6f}")


class CallbackList:
    """
    Gestionnaire de liste de callbacks.
    """
    
    def __init__(self, callbacks=None):
        """
        Args:
            callbacks: Liste de callbacks
        """
        self.callbacks = callbacks or []
    
    def add(self, callback):
        """Ajoute un callback."""
        self.callbacks.append(callback)
    
    def on_training_begin(self, agent, env):
        """Appelle tous les callbacks."""
        for callback in self.callbacks:
            callback.on_training_begin(agent, env)
    
    def on_training_end(self, agent, env):
        """Appelle tous les callbacks."""
        for callback in self.callbacks:
            callback.on_training_end(agent, env)
    
    def on_episode_begin(self, episode, agent, env):
        """Appelle tous les callbacks."""
        for callback in self.callbacks:
            callback.on_episode_begin(episode, agent, env)
    
    def on_episode_end(self, episode, agent, env, episode_info):
        """Appelle tous les callbacks."""
        should_stop = False
        for callback in self.callbacks:
            result = callback.on_episode_end(episode, agent, env, episode_info)
            if result is True:  # Early stopping
                should_stop = True
        return should_stop
    
    def on_step(self, episode, step, agent, state, action, reward, next_state, done):
        """Appelle tous les callbacks."""
        for callback in self.callbacks:
            callback.on_step(episode, step, agent, state, action, reward, next_state, done)


print(" Système de callbacks chargé avec succès!")
