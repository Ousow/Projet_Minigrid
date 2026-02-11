"""
Script de Comparaison Automatique des Agents RL
================================================

Ce script permet de comparer facilement différentes configurations d'agents
et de générer des rapports de comparaison détaillés.
"""

import gymnasium as gym
import minigrid
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path


class AgentComparator:
    """
    Classe pour comparer plusieurs agents de manière systématique.
    """
    
    def __init__(self, env_name='MiniGrid-Empty-8x8-v0', output_dir='comparisons'):
        """
        Initialise le comparateur.
        
        Args:
            env_name: Nom de l'environnement à utiliser
            output_dir: Dossier pour sauvegarder les résultats
        """
        self.env_name = env_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.experiments = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_experiment(self, name, agent_class, agent_params, 
                       training_params, num_runs=5):
        """
        Ajoute une expérience à comparer.
        
        Args:
            name: Nom de l'expérience
            agent_class: Classe de l'agent
            agent_params: Paramètres pour initialiser l'agent
            training_params: Paramètres d'entraînement
            num_runs: Nombre de runs avec différentes seeds
        """
        self.experiments.append({
            'name': name,
            'agent_class': agent_class,
            'agent_params': agent_params,
            'training_params': training_params,
            'num_runs': num_runs,
            'results': []
        })
    
    def run_comparison(self):
        """
        Exécute toutes les expériences et collecte les résultats.
        """
        print(f"\n{'='*70}")
        print(f"COMPARAISON DES AGENTS SUR {self.env_name}")
        print(f"{'='*70}\n")
        
        for exp_idx, experiment in enumerate(self.experiments):
            print(f"\n[{exp_idx+1}/{len(self.experiments)}] Expérience: {experiment['name']}")
            print("-" * 70)
            
            for run in range(experiment['num_runs']):
                print(f"  Run {run+1}/{experiment['num_runs']}...", end=' ')
                
                # Créer l'environnement
                env = gym.make(self.env_name)
                
                # Initialiser l'agent
                agent = experiment['agent_class'](**experiment['agent_params'])
                
                # Entraîner
                seed = 42 + run
                np.random.seed(seed)
                history = agent.train(env, **experiment['training_params'], verbose=False)
                
                # Évaluer
                eval_results = agent.evaluate(env, num_episodes=100, max_steps=500)
                
                # Stocker les résultats
                experiment['results'].append({
                    'seed': seed,
                    'training_history': history,
                    'evaluation': eval_results
                })
                
                env.close()
                print(f"✓ (Succès: {eval_results['success_rate']*100:.1f}%)")
            
            print(f"  Moyenne sur {experiment['num_runs']} runs:")
            self._print_summary(experiment)
    
    def _print_summary(self, experiment):
        """Affiche un résumé des résultats d'une expérience."""
        rewards = [r['evaluation']['mean_reward'] for r in experiment['results']]
        steps = [r['evaluation']['mean_steps'] for r in experiment['results']]
        success_rates = [r['evaluation']['success_rate'] for r in experiment['results']]
        
        print(f"    Récompense: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
        print(f"    Étapes: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
        print(f"    Taux succès: {np.mean(success_rates)*100:.1f}% ± {np.std(success_rates)*100:.1f}%")
    
    def generate_comparison_plots(self):
        """
        Génère des graphiques de comparaison.
        """
        print(f"\n{'='*70}")
        print("GÉNÉRATION DES GRAPHIQUES DE COMPARAISON")
        print(f"{'='*70}\n")
        
        # 1. Courbes d'apprentissage moyennes
        self._plot_learning_curves()
        
        # 2. Comparaison des performances finales
        self._plot_final_performance()
        
        # 3. Boxplots des résultats
        self._plot_boxplots()
        
        # 4. Analyse de convergence
        self._plot_convergence_analysis()
    
    def _plot_learning_curves(self):
        """Trace les courbes d'apprentissage moyennes."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        for experiment in self.experiments:
            # Calculer la moyenne sur tous les runs
            all_rewards = [r['training_history']['rewards'] for r in experiment['results']]
            
            # Trouver la longueur minimale
            min_length = min(len(r) for r in all_rewards)
            all_rewards = [r[:min_length] for r in all_rewards]
            
            mean_rewards = np.mean(all_rewards, axis=0)
            std_rewards = np.std(all_rewards, axis=0)
            
            episodes = range(len(mean_rewards))
            
            # Récompenses
            axes[0].plot(episodes, mean_rewards, label=experiment['name'], linewidth=2)
            axes[0].fill_between(episodes, 
                                mean_rewards - std_rewards,
                                mean_rewards + std_rewards,
                                alpha=0.2)
            
            # Étapes
            all_steps = [r['training_history']['steps'] for r in experiment['results']]
            all_steps = [s[:min_length] for s in all_steps]
            mean_steps = np.mean(all_steps, axis=0)
            std_steps = np.std(all_steps, axis=0)
            
            axes[1].plot(episodes, mean_steps, label=experiment['name'], linewidth=2)
            axes[1].fill_between(episodes,
                                mean_steps - std_steps,
                                mean_steps + std_steps,
                                alpha=0.2)
        
        axes[0].set_xlabel('Épisode')
        axes[0].set_ylabel('Récompense moyenne')
        axes[0].set_title('Courbes d\'apprentissage - Récompenses')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Épisode')
        axes[1].set_ylabel('Nombre d\'étapes moyen')
        axes[1].set_title('Courbes d\'apprentissage - Étapes')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f'learning_curves_{self.timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Courbes d'apprentissage sauvegardées: {save_path}")
        plt.close()
    
    def _plot_final_performance(self):
        """Compare les performances finales."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        names = [exp['name'] for exp in self.experiments]
        x_pos = np.arange(len(names))
        
        # Récompenses
        rewards_mean = []
        rewards_std = []
        for exp in self.experiments:
            rewards = [r['evaluation']['mean_reward'] for r in exp['results']]
            rewards_mean.append(np.mean(rewards))
            rewards_std.append(np.std(rewards))
        
        axes[0].bar(x_pos, rewards_mean, yerr=rewards_std, 
                   capsize=5, color='skyblue', edgecolor='black')
        axes[0].set_ylabel('Récompense moyenne')
        axes[0].set_title('Récompenses en évaluation')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(names, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Étapes
        steps_mean = []
        steps_std = []
        for exp in self.experiments:
            steps = [r['evaluation']['mean_steps'] for r in exp['results']]
            steps_mean.append(np.mean(steps))
            steps_std.append(np.std(steps))
        
        axes[1].bar(x_pos, steps_mean, yerr=steps_std,
                   capsize=5, color='lightgreen', edgecolor='black')
        axes[1].set_ylabel('Nombre d\'étapes moyen')
        axes[1].set_title('Étapes en évaluation')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(names, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Taux de succès
        success_mean = []
        success_std = []
        for exp in self.experiments:
            success = [r['evaluation']['success_rate'] * 100 for r in exp['results']]
            success_mean.append(np.mean(success))
            success_std.append(np.std(success))
        
        axes[2].bar(x_pos, success_mean, yerr=success_std,
                   capsize=5, color='lightcoral', edgecolor='black')
        axes[2].set_ylabel('Taux de succès (%)')
        axes[2].set_title('Taux de succès en évaluation')
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(names, rotation=45, ha='right')
        axes[2].set_ylim([0, 105])
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.output_dir / f'final_performance_{self.timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Performances finales sauvegardées: {save_path}")
        plt.close()
    
    def _plot_boxplots(self):
        """Crée des boxplots pour chaque métrique."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Préparer les données
        all_rewards = []
        all_steps = []
        all_success = []
        labels = []
        
        for exp in self.experiments:
            rewards = [r['evaluation']['mean_reward'] for r in exp['results']]
            steps = [r['evaluation']['mean_steps'] for r in exp['results']]
            success = [r['evaluation']['success_rate'] * 100 for r in exp['results']]
            
            all_rewards.append(rewards)
            all_steps.append(steps)
            all_success.append(success)
            labels.append(exp['name'])
        
        # Boxplots
        axes[0].boxplot(all_rewards, labels=labels)
        axes[0].set_ylabel('Récompense moyenne')
        axes[0].set_title('Distribution des récompenses')
        axes[0].grid(True, alpha=0.3, axis='y')
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        axes[1].boxplot(all_steps, labels=labels)
        axes[1].set_ylabel('Nombre d\'étapes moyen')
        axes[1].set_title('Distribution des étapes')
        axes[1].grid(True, alpha=0.3, axis='y')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        axes[2].boxplot(all_success, labels=labels)
        axes[2].set_ylabel('Taux de succès (%)')
        axes[2].set_title('Distribution du taux de succès')
        axes[2].grid(True, alpha=0.3, axis='y')
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        save_path = self.output_dir / f'boxplots_{self.timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Boxplots sauvegardés: {save_path}")
        plt.close()
    
    def _plot_convergence_analysis(self):
        """Analyse la vitesse de convergence."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for exp in self.experiments:
            # Moyenne des courbes d'apprentissage
            all_rewards = [r['training_history']['rewards'] for r in exp['results']]
            min_length = min(len(r) for r in all_rewards)
            all_rewards = [r[:min_length] for r in all_rewards]
            mean_rewards = np.mean(all_rewards, axis=0)
            
            # Trouver le point de convergence (90% de la performance finale)
            final_perf = np.mean(mean_rewards[-100:])
            threshold = 0.9 * final_perf
            
            convergence_point = None
            for i, reward in enumerate(mean_rewards):
                if reward >= threshold:
                    convergence_point = i
                    break
            
            # Tracer
            episodes = range(len(mean_rewards))
            ax.plot(episodes, mean_rewards, label=exp['name'], linewidth=2)
            
            if convergence_point:
                ax.axvline(convergence_point, linestyle='--', alpha=0.5)
                ax.text(convergence_point, threshold, 
                       f"{convergence_point} ép.", 
                       rotation=90, va='bottom')
        
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Récompense moyenne')
        ax.set_title('Analyse de la convergence (90% de la performance finale)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f'convergence_{self.timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Analyse de convergence sauvegardée: {save_path}")
        plt.close()
    
    def export_results(self):
        """Exporte les résultats en JSON."""
        results_summary = []
        
        for exp in self.experiments:
            rewards = [r['evaluation']['mean_reward'] for r in exp['results']]
            steps = [r['evaluation']['mean_steps'] for r in exp['results']]
            success = [r['evaluation']['success_rate'] for r in exp['results']]
            
            results_summary.append({
                'name': exp['name'],
                'agent_params': exp['agent_params'],
                'training_params': exp['training_params'],
                'num_runs': exp['num_runs'],
                'results': {
                    'reward_mean': float(np.mean(rewards)),
                    'reward_std': float(np.std(rewards)),
                    'steps_mean': float(np.mean(steps)),
                    'steps_std': float(np.std(steps)),
                    'success_rate_mean': float(np.mean(success)),
                    'success_rate_std': float(np.std(success))
                }
            })
        
        save_path = self.output_dir / f'results_{self.timestamp}.json'
        with open(save_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"✓ Résultats exportés: {save_path}")


# Exemple d'utilisation
if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║              COMPARATEUR D'AGENTS RL - MINIGRID                  ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    
    Ce script permet de comparer systématiquement différentes configurations
    d'agents sur MiniGrid.
    
    UTILISATION:
    
    1. Importez vos classes d'agents
    2. Définissez les expériences à comparer
    3. Exécutez la comparaison
    4. Générez les visualisations
    
    EXEMPLE:
    
    from 2_qlearning_agent import QLearningAgent
    
    comparator = AgentComparator('MiniGrid-Empty-8x8-v0')
    
    # Expérience 1: Q-Learning avec learning rate 0.1
    comparator.add_experiment(
        name='Q-Learning (lr=0.1)',
        agent_class=QLearningAgent,
        agent_params={'action_space_size': 7, 'learning_rate': 0.1},
        training_params={'num_episodes': 500, 'max_steps': 500},
        num_runs=5
    )
    
    # Expérience 2: Q-Learning avec learning rate 0.5
    comparator.add_experiment(
        name='Q-Learning (lr=0.5)',
        agent_class=QLearningAgent,
        agent_params={'action_space_size': 7, 'learning_rate': 0.5},
        training_params={'num_episodes': 500, 'max_steps': 500},
        num_runs=5
    )
    
    # Exécuter
    comparator.run_comparison()
    comparator.generate_comparison_plots()
    comparator.export_results()
    
    Les résultats seront dans le dossier 'comparisons/'.
    """)
