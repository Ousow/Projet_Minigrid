"""
VISUALISATION DES RÉSULTATS - SANS TENSORBOARD !
📊 Graphiques automatiques à partir des callbacks
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualiser_tout():
    """Génère TOUS les graphiques des callbacks"""
    
    print("\n" + "="*60)
    print("📊 VISUALISATION DES RÉSULTATS D'ENTRAÎNEMENT")
    print("="*60)
    
    # Chercher les dossiers runs
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("❌ Aucun dossier 'runs' trouvé !")
        return
    
    # Trouver la dernière run
    runs = list(runs_dir.glob("dqn_*"))
    if not runs:
        print("❌ Aucune run trouvée dans runs/")
        return
    
    latest_run = sorted(runs)[-1]
    print(f"\n📁 Dernière run trouvée : {latest_run.name}")
    
    # Charger les métriques
    metrics_file = latest_run / "metrics.json"
    if not metrics_file.exists():
        print("❌ Fichier metrics.json non trouvé !")
        return
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Créer les graphiques
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Résultats d'entraînement - {latest_run.name}", fontsize=16, fontweight='bold')
    
    # 1. Courbe des récompenses
    ax = axes[0, 0]
    rewards = metrics.get('rewards', [])
    episodes = range(1, len(rewards) + 1)
    
    ax.plot(episodes, rewards, 'b-', alpha=0.5, linewidth=0.5, label='Reward brut')
    
    # Moyenne mobile
    if len(rewards) > 50:
        window = 50
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(rewards)+1), moving_avg, 'r-', linewidth=2, label=f'Moyenne mobile ({window})')
    
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Récompense')
    ax.set_title('🎯 Récompenses par épisode', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Ajouter la valeur max
    if rewards:
        max_reward = max(rewards)
        max_ep = rewards.index(max_reward) + 1
        ax.plot(max_ep, max_reward, 'go', markersize=10, label=f'Max: {max_reward:.3f}')
        ax.legend()
    
    # 2. Courbe d'epsilon
    ax = axes[0, 1]
    epsilons = metrics.get('epsilons', [])
    if epsilons:
        ax.plot(range(1, len(epsilons)+1), epsilons, 'g-', linewidth=2)
        ax.set_xlabel('Épisode')
        ax.set_ylabel('Epsilon')
        ax.set_title('🔄 Exploration (Epsilon)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
    
    # 3. Courbe des pertes (loss)
    ax = axes[1, 0]
    losses = metrics.get('losses', [])
    if losses:
        ax.plot(range(1, len(losses)+1), losses, 'purple', alpha=0.5, linewidth=0.5)
        
        # Moyenne mobile pour loss
        if len(losses) > 50:
            window = 50
            moving_avg_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(losses)+1), moving_avg_loss, 'orange', linewidth=2, label=f'Moyenne mobile ({window})')
        
        ax.set_xlabel('Update step')
        ax.set_ylabel('Loss')
        ax.set_title('📉 Loss (MSE)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # 4. Évaluation finale
    ax = axes[1, 1]
    
    # Chercher le meilleur modèle
    best_model_path = Path("models/dqn_minigrid/best_model.json")
    if best_model_path.exists():
        with open(best_model_path, 'r') as f:
            best_data = json.load(f)
        
        eval_reward = best_data.get('metrics', {}).get('eval_reward', 0)
        success_rate = best_data.get('metrics', {}).get('success_rate', 0)
        
        # Bar chart des performances
        categories = ['Reward', 'Succès %']
        values = [eval_reward, success_rate * 100]
        colors = ['#2ecc71', '#3498db']
        
        bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=2)
        ax.set_ylim([0, 110])
        ax.set_title('🏆 Performance finale', fontweight='bold')
        ax.set_ylabel('Valeur')
        
        # Ajouter les valeurs sur les barres
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{val:.1f}{"%" if "Succès" in bar.get_label() else ""}',
                   ha='center', va='bottom', fontweight='bold', fontsize=12)
    else:
        ax.text(0.5, 0.5, "Modèle non trouvé", ha='center', va='center', transform=ax.transAxes)
        ax.set_title('⚠️ Performance non disponible', fontweight='bold')
    
    plt.tight_layout()
    
    # Sauvegarder le graphique
    output_file = f"resultats_{latest_run.name}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Graphique sauvegardé : {output_file}")
    
    # Afficher
    plt.show()
    
    # Afficher un résumé texte
    print("\n" + "="*60)
    print("📈 RÉSUMÉ DES PERFORMANCES")
    print("="*60)
    print(f"\n🎯 Récompense maximale : {max(rewards):.3f}")
    print(f"📉 Récompense moyenne (fin) : {np.mean(rewards[-100:]):.3f}")
    print(f"🔄 Epsilon final : {epsilons[-1] if epsilons else 0:.3f}")
    
    if best_model_path.exists():
        with open(best_model_path, 'r') as f:
            best_data = json.load(f)
        print(f"\n🏆 MEILLEUR MODÈLE (épisode {best_data.get('epoch', '?')})")
        print(f"   - Reward évaluation : {best_data.get('metrics', {}).get('eval_reward', 0):.3f}")
        print(f"   - Taux de succès : {best_data.get('metrics', {}).get('success_rate', 0)*100:.1f}%")
    
    print("\n" + "="*60)
    print("✅ VISUALISATION TERMINÉE !")
    print("="*60)

def visualiser_comparaison():
    """Compare plusieurs runs"""
    
    runs_dir = Path("runs")
    runs = sorted(runs_dir.glob("dqn_*"))
    
    if len(runs) < 2:
        print("Pas assez de runs pour comparer")
        return
    
    plt.figure(figsize=(12, 6))
    
    for run in runs[-3:]:  # 3 dernières runs
        metrics_file = run / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            rewards = metrics.get('rewards', [])
            if rewards:
                window = 50
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                plt.plot(range(window, len(rewards)+1), moving_avg, 
                        linewidth=2, label=run.name)
    
    plt.xlabel('Épisode')
    plt.ylabel('Récompense (moyenne mobile)')
    plt.title('📊 Comparaison des différentes runs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('comparaison_runs.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔍 ANALYSE DES CALLBACKS DQN MINIGRID")
    print("="*60)
    
    visualiser_tout()
    
    # Décommenter pour comparer plusieurs runs
    # visualiser_comparaison()