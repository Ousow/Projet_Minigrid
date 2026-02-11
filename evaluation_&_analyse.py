
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from pathlib import Path

print("="*70)
print("GÉNÉRATION DES GRAPHIQUES À PARTIR DES AGENTS ENTRAÎNÉS")
print("="*70)

# Créer le dossier results
Path("results").mkdir(exist_ok=True)

# CHARGER LES RÉSULTATS DES AGENTS
print("\n Chargement des agents entraînés...")

# Charger Q-Learning
try:
    with open('qlearning_agent.pkl', 'rb') as f:
        qlearning_data = pickle.load(f)
    
    if isinstance(qlearning_data, dict):
        # Format : dictionnaire avec les données
        q_rewards = qlearning_data.get('training_rewards', [])
        q_steps = qlearning_data.get('training_steps', [])
        q_eval = {
            'mean_reward': qlearning_data.get('eval_mean_reward', 0),
            'std_reward': qlearning_data.get('eval_std_reward', 0),
            'mean_steps': qlearning_data.get('eval_mean_steps', 0),
            'std_steps': qlearning_data.get('eval_std_steps', 0),
            'success_rate': qlearning_data.get('eval_success_rate', 0),
        }
    else:
        # Format : objet agent directement
        q_rewards = qlearning_data.training_rewards
        q_steps = qlearning_data.training_steps
        q_eval = None  # Pas de données d'éval
    
    print(f" Q-Learning chargé : {len(q_rewards)} épisodes")
    has_qlearning = True
except FileNotFoundError:
    print(" qlearning_agent.pkl non trouvé")
    has_qlearning = False
except Exception as e:
    print(f" Erreur chargement Q-Learning : {e}")
    has_qlearning = False

# Charger DQN
try:
    checkpoint = torch.load('dqn_agent.pth', 
                       map_location='cpu',
                       weights_only=False)
    
    dqn_rewards = checkpoint.get('training_rewards', [])
    dqn_steps = checkpoint.get('training_steps', [])
    dqn_losses = checkpoint.get('losses', [])
    dqn_eval = None  # Pas de données d'éval dans le checkpoint
    
    print(f" DQN chargé : {len(dqn_rewards)} épisodes")
    has_dqn = True
except FileNotFoundError:
    print(" dqn_agent.pth non trouvé")
    has_dqn = False
except Exception as e:
    print(f" Erreur chargement DQN : {e}")
    has_dqn = False

if not has_qlearning and not has_dqn:
    print("\n ERREUR : Aucun agent trouvé !")
    print("\nVérifiez que vous avez bien :")
    print("  - qlearning_agent.pkl")
    print("  - dqn_agent.pth")
    print("\ndans le même dossier que ce script.")
    exit(1)

# GRAPHIQUE 1 : COURBES D'APPRENTISSAGE

print("\n Génération des graphiques...")

num_agents = sum([has_qlearning, has_dqn])
fig, axes = plt.subplots(2, num_agents, figsize=(8*num_agents, 10))

if num_agents == 1:
    axes = axes.reshape(-1, 1)

col = 0

# Q-Learning
if has_qlearning:
    window = 100
    
    # Récompenses
    ax = axes[0, col]
    episodes = range(1, len(q_rewards) + 1)
    moving_avg = np.convolve(q_rewards, np.ones(window)/window, mode='valid')
    
    ax.plot(episodes, q_rewards, alpha=0.3, color='blue', label='Récompense brute')
    ax.plot(range(window, len(q_rewards)+1), moving_avg, 
           color='red', linewidth=2, label=f'Moyenne mobile ({window})')
    ax.set_xlabel('Épisode', fontsize=12)
    ax.set_ylabel('Récompense', fontsize=12)
    ax.set_title('Q-Learning - Récompenses', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Étapes
    ax = axes[1, col]
    moving_avg_steps = np.convolve(q_steps, np.ones(window)/window, mode='valid')
    
    ax.plot(episodes, q_steps, alpha=0.3, color='green', label='Étapes brutes')
    ax.plot(range(window, len(q_steps)+1), moving_avg_steps,
           color='orange', linewidth=2, label=f'Moyenne mobile ({window})')
    ax.set_xlabel('Épisode', fontsize=12)
    ax.set_ylabel('Nombre d\'étapes', fontsize=12)
    ax.set_title('Q-Learning - Étapes par épisode', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    col += 1

# DQN
if has_dqn:
    window = 100
    
    # Récompenses
    ax = axes[0, col]
    episodes = range(1, len(dqn_rewards) + 1)
    moving_avg = np.convolve(dqn_rewards, np.ones(window)/window, mode='valid')
    
    ax.plot(episodes, dqn_rewards, alpha=0.3, color='blue', label='Récompense brute')
    ax.plot(range(window, len(dqn_rewards)+1), moving_avg,
           color='red', linewidth=2, label=f'Moyenne mobile ({window})')
    ax.set_xlabel('Épisode', fontsize=12)
    ax.set_ylabel('Récompense', fontsize=12)
    ax.set_title('DQN - Récompenses', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Étapes
    ax = axes[1, col]
    moving_avg_steps = np.convolve(dqn_steps, np.ones(window)/window, mode='valid')
    
    ax.plot(episodes, dqn_steps, alpha=0.3, color='green', label='Étapes brutes')
    ax.plot(range(window, len(dqn_steps)+1), moving_avg_steps,
           color='orange', linewidth=2, label=f'Moyenne mobile ({window})')
    ax.set_xlabel('Épisode', fontsize=12)
    ax.set_ylabel('Nombre d\'étapes', fontsize=12)
    ax.set_title('DQN - Étapes par épisode', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
print("✓ results/training_curves.png")
plt.close()

# GRAPHIQUE 2 : COMPARAISON DES PERFORMANCES
if has_qlearning and has_dqn:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Récompenses finales (derniers 100 épisodes)
    ax = axes[0]
    q_final = np.mean(q_rewards[-100:])
    dqn_final = np.mean(dqn_rewards[-100:])
    
    bars = ax.bar(['Q-Learning', 'DQN'], [q_final, dqn_final], 
                  color=['skyblue', 'lightcoral'], edgecolor='black', width=0.6)
    ax.set_ylabel('Récompense moyenne (100 derniers ép.)', fontsize=12)
    ax.set_title('Comparaison des récompenses finales', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs sur les barres
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Étapes finales
    ax = axes[1]
    q_steps_final = np.mean(q_steps[-100:])
    dqn_steps_final = np.mean(dqn_steps[-100:])
    
    bars = ax.bar(['Q-Learning', 'DQN'], [q_steps_final, dqn_steps_final],
                  color=['lightgreen', 'orange'], edgecolor='black', width=0.6)
    ax.set_ylabel('Nombre d\'étapes moyen (100 derniers ép.)', fontsize=12)
    ax.set_title('Comparaison du nombre d\'étapes', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Ajouter les valeurs sur les barres
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ results/performance_comparison.png")
    plt.close()

# GRAPHIQUE 3 : ÉVOLUTION DE LA CONVERGENCE

fig, ax = plt.subplots(figsize=(12, 6))

if has_qlearning:
    window = 100
    moving_avg = np.convolve(q_rewards, np.ones(window)/window, mode='valid')
    ax.plot(range(window, len(q_rewards)+1), moving_avg, 
           linewidth=2.5, label='Q-Learning', color='blue')

if has_dqn:
    window = 100
    moving_avg = np.convolve(dqn_rewards, np.ones(window)/window, mode='valid')
    ax.plot(range(window, len(dqn_rewards)+1), moving_avg,
           linewidth=2.5, label='DQN', color='red')

ax.set_xlabel('Épisode', fontsize=12)
ax.set_ylabel('Récompense moyenne mobile', fontsize=12)
ax.set_title('Comparaison de la convergence des algorithmes', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/convergence_comparison.png', dpi=300, bbox_inches='tight')
print("✓ results/convergence_comparison.png")
plt.close()

# GRAPHIQUE 4 : LOSS DQN 
if has_dqn and dqn_losses:
    fig, ax = plt.subplots(figsize=(12, 5))
    
    window = 50
    if len(dqn_losses) > window:
        moving_avg = np.convolve(dqn_losses, np.ones(window)/window, mode='valid')
        episodes = range(1, len(dqn_losses) + 1)
        
        ax.plot(episodes, dqn_losses, alpha=0.3, color='purple', label='Loss brute')
        ax.plot(range(window, len(dqn_losses)+1), moving_avg,
               color='darkviolet', linewidth=2, label=f'Moyenne mobile ({window})')
    else:
        ax.plot(dqn_losses, color='purple')
    
    ax.set_xlabel('Update', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('DQN - Évolution de la Loss', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/dqn_loss.png', dpi=300, bbox_inches='tight')
    print("✓ results/dqn_loss.png")
    plt.close()

# RAPPORT TEXTUEL

print("\n Génération du rapport...")

with open('results/performance_report.txt', 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("RAPPORT DE PERFORMANCE - APPRENTISSAGE PAR RENFORCEMENT\n")
    f.write("="*70 + "\n\n")
    
    if has_qlearning:
        f.write("Q-LEARNING\n")
        f.write("-"*70 + "\n")
        f.write(f"Nombre d'épisodes: {len(q_rewards)}\n")
        f.write(f"Récompense moyenne (premiers 100 ép.): {np.mean(q_rewards[:100]):.3f}\n")
        f.write(f"Récompense moyenne (derniers 100 ép.): {np.mean(q_rewards[-100:]):.3f}\n")
        f.write(f"Récompense maximale: {np.max(q_rewards):.3f}\n")
        f.write(f"Récompense minimale: {np.min(q_rewards):.3f}\n")
        f.write(f"\nÉtapes moyennes (premiers 100 ép.): {np.mean(q_steps[:100]):.1f}\n")
        f.write(f"Étapes moyennes (derniers 100 ép.): {np.mean(q_steps[-100:]):.1f}\n")
        f.write("\n")
    
    if has_dqn:
        f.write("DEEP Q-NETWORK (DQN)\n")
        f.write("-"*70 + "\n")
        f.write(f"Nombre d'épisodes: {len(dqn_rewards)}\n")
        f.write(f"Récompense moyenne (premiers 100 ép.): {np.mean(dqn_rewards[:100]):.3f}\n")
        f.write(f"Récompense moyenne (derniers 100 ép.): {np.mean(dqn_rewards[-100:]):.3f}\n")
        f.write(f"Récompense maximale: {np.max(dqn_rewards):.3f}\n")
        f.write(f"Récompense minimale: {np.min(dqn_rewards):.3f}\n")
        f.write(f"\nÉtapes moyennes (premiers 100 ép.): {np.mean(dqn_steps[:100]):.1f}\n")
        f.write(f"Étapes moyennes (derniers 100 ép.): {np.mean(dqn_steps[-100:]):.1f}\n")
        f.write("\n")
    
    if has_qlearning and has_dqn:
        f.write("COMPARAISON\n")
        f.write("-"*70 + "\n")
        q_final = np.mean(q_rewards[-100:])
        dqn_final = np.mean(dqn_rewards[-100:])
        
        if q_final > dqn_final:
            f.write(f"Q-Learning obtient de meilleures récompenses finales\n")
            f.write(f"Différence: +{q_final - dqn_final:.3f}\n")
        else:
            f.write(f"DQN obtient de meilleures récompenses finales\n")
            f.write(f"Différence: +{dqn_final - q_final:.3f}\n")

print("✓ results/performance_report.txt")

# RÉSUMÉ FINAL
print("\n" + "="*70)
print("RÉSUMÉ DES RÉSULTATS")
print("="*70 + "\n")

if has_qlearning:
    print("Q-LEARNING:")
    print(f"   Récompense finale: {np.mean(q_rewards[-100:]):.3f}")
    print(f"   Étapes finales: {np.mean(q_steps[-100:]):.1f}")
    print(f"   Amélioration: {np.mean(q_rewards[-100:]) - np.mean(q_rewards[:100]):.3f}")
    print()

if has_dqn:
    print("DQN:")
    print(f"   Récompense finale: {np.mean(dqn_rewards[-100:]):.3f}")
    print(f"   Étapes finales: {np.mean(dqn_steps[-100:]):.1f}")
    print(f"   Amélioration: {np.mean(dqn_rewards[-100:]) - np.mean(dqn_rewards[:100]):.3f}")
    print()

print("="*70)
print(" GÉNÉRATION TERMINÉE AVEC SUCCÈS !")
print("="*70)
print("\n Tous vos graphiques sont dans le dossier 'results/' :")
print("   - training_curves.png")
if has_qlearning and has_dqn:
    print("   - performance_comparison.png")
    print("   - convergence_comparison.png")
if has_dqn and dqn_losses:
    print("   - dqn_loss.png")
print("   - performance_report.txt")
print("\n Vous pouvez maintenant :")
print("   1. Ouvrir ces images pour analyser vos résultats")
print("   2. Les insérer dans votre rapport_tp.md")
print("   3. Analyser les performances de vos agents")
print("="*70 + "\n")
