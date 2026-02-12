"""
TP Apprentissage par Renforcement
Avec DQN CNN + Callbacks
"""

import gymnasium as gym
import minigrid
import subprocess
import sys
from pathlib import Path

def print_section(title):
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")

def run_dqn_cnn():
    """Lance TON DQN avec CNN et callbacks"""
    print_section("🚀 DQN AMÉLIORÉ - CNN + CALLBACKS")
    print("Lancement de ton agent avec:")
    print("  ✅ CNN profond pour perception spatiale")
    print("  ✅ TensorBoard pour monitoring")
    print("  ✅ ModelCheckpoint pour sauvegarde auto")
    print("  ✅ EarlyStopping pour arrêt intelligent")
    print("\n" + "-"*50)
    
    subprocess.run([sys.executable, "dqn_cnn.py"])

def run_comparison():
    """Lance le benchmark comparatif"""
    print_section("📊 BENCHMARK COMPARATIF")
    print("Comparaison des performances:")
    print("  - DQN baseline (camarade)")
    print("  - DQN CNN amélioré (TOI)")
    print("\n" + "-"*50)
    
    subprocess.run([sys.executable, "comparison_benchmark.py"])

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║     TP APPRENTISSAGE PAR RENFORCEMENT - MINIGRID        ║
║              VERSION AMÉLIORÉE - TES AJOUTS             ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    while True:
        print("\n📋 MENU PRINCIPAL:")
        print("  ──────────────────────────────")
        print("  🚀  1. Lancer TON DQN CNN (callbacks)")
        print("  📊  2. Lancer benchmark comparatif")
        print("  📈  3. Lancer TensorBoard")
        print("  📁  4. Voir structure projet")
        print("  ❌  5. Quitter")
        print("  ──────────────────────────────")
        
        choice = input("\n👉 Ton choix (1-5): ").strip()
        
        if choice == '1':
            run_dqn_cnn()
        elif choice == '2':
            run_comparison()
        elif choice == '3':
            print("\n📊 Lancement TensorBoard...")
            print("   URL: http://localhost:6006")
            print("   Appuie sur CTRL+C pour arrêter\n")
            subprocess.run(["tensorboard", "--logdir=runs"])
        elif choice == '4':
            print_section("STRUCTURE DU PROJET")
            print("""
📁 Projet_Minigrid/
├── 📄 dqn_cnn.py              # ✅ TON DQN avec CNN + callbacks
├── 📄 dqn_agent.py            # 📕 DQN baseline (camarade)
├── 📄 qlearning_agent.py      # 📕 Q-Learning (camarade)
├── 📄 comparison_benchmark.py # ✅ TON benchmark
├── 📄 main_ameliore.py        # ✅ CE MENU
│
├── 📁 callbacks/              # ✅ TES callbacks
│   ├── tensorboard_callback.py
│   ├── model_checkpoint.py
│   └── early_stopping.py
│
├── 📁 runs/                   # 📊 Logs TensorBoard
├── 📁 models/                 # 💾 Modèles sauvegardés
└── 📁 results/                # 📈 Graphiques
            """)
        elif choice == '5':
            print("\n👋 Au revoir !\n")
            break
        else:
            print("\n❌ Choix invalide. Choisis 1-5.")

if __name__ == "__main__":
    main()



"""
TP Apprentissage par Renforcement avec MiniGrid - Script Principal

Ce script orchestre l'ensemble du TP:
1. Exploration de MiniGrid
2. Entraînement des agents (Q-Learning et DQN)
3. Évaluation et analyse des résultats
4. Génération du rapport
"""

import gymnasium as gym
import minigrid
import numpy as np
import sys
from pathlib import Path

# Imports des modules du TP
def print_section(title):
    """Affiche un titre de section."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def run_exploration():
    """Exécute la partie exploration."""
    print_section("EXPLORATION DE MINIGRID")
    
    print("Pour exécuter l'exploration, lancez:")
    print("  python exploration_minigrid.py")
    print("\nCette partie vous permettra de:")
    print("  - Comprendre l'espace d'observations")
    print("  - Découvrir les actions disponibles")
    print("  - Analyser le système de récompenses")


def run_qlearning_training():
    """Exécute l'entraînement Q-Learning."""
    print_section("ENTRAÎNEMENT Q-LEARNING")
    
    print("Pour entraîner l'agent Q-Learning, lancez:")
    print("  python qlearning_agent.py")
    print("\nCet agent utilise:")
    print("  - Table Q tabulaire")
    print("  - Exploration epsilon-greedy")
    print("  - Mise à jour par différence temporelle")
    print("\nDurée estimée: 2-5 minutes")


def run_dqn_training():
    """Exécute l'entraînement DQN."""
    print_section("ENTRAÎNEMENT DQN")
    
    print("Pour entraîner l'agent DQN, lancez:")
    print("  python dqn_agent.py")
    print("\nCet agent utilise:")
    print("  - Réseau de neurones profond")
    print("  - Replay buffer")
    print("  - Target network")
    print("\nDurée estimée: 5-10 minutes (CPU) ou 2-3 minutes (GPU)")


def run_evaluation():
    """Exécute l'évaluation et génère les visualisations."""
    print_section("ÉVALUATION ET ANALYSE")
    
    print("Pour analyser les résultats, lancez:")
    print("  python evaluation_analysis.py")
    print("\nCette partie génère:")
    print("  - Courbes d'apprentissage")
    print("  - Comparaison des performances")
    print("  - Distribution des récompenses")
    print("  - Rapport textuel détaillé")


def check_dependencies():
    """Vérifie que les dépendances sont installées."""
    print_section("VÉRIFICATION DES DÉPENDANCES")
    
    dependencies = {
        'gymnasium': 'gymnasium',
        'minigrid': 'minigrid',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib.pyplot',
        'torch': 'torch',
        'tqdm': 'tqdm'
    }
    
    missing = []
    
    for name, import_name in dependencies.items():
        try:
            __import__(import_name)
            print(f"✓ {name} installé")
        except ImportError:
            print(f"✗ {name} manquant")
            missing.append(name)
    
    if missing:
        print(f"\nDépendances manquantes: {', '.join(missing)}")
        print("\nPour les installer:")
        print("  pip install gymnasium minigrid numpy matplotlib torch tqdm")
        return False
    
    print("\n✓ Toutes les dépendances sont installées!")
    return True


def show_project_structure():
    """Affiche la structure du projet."""
    print_section("STRUCTURE DU PROJET")
    
    structure = """
tp_minigrid/
│
├── exploration_minigrid.py    # Découverte de l'environnement
├── qlearning_agent.py         # Agent Q-Learning
├── dqn_agent.py               # Agent Deep Q-Network
├── evaluation_&_analyse.py     # Évaluation et visualisation
├── main.py                      # Ce script
├── requirements.txt             # Dépendances
│
├── results/                     # Résultats générés
│   ├── training_curves.png
│   ├── performance_comparison.png
|   ├── convergence_comparaison.png
│   ├── dqn_loss.png
│   └── performance_report.txt
│
├── qlearning_agent.pkl          # Agent Q-Learning sauvegardé
├── dqn_agent.pth                # Agent DQN sauvegardé
│
└── rapport_tp.md                # Rapport final
    """
    
    print(structure)


def show_workflow():
    """Affiche le workflow recommandé."""
    print_section("WORKFLOW RECOMMANDÉ")
    
    workflow = """
ÉTAPE 1: EXPLORATION 
  → Exécuter: python exploration_minigrid.py
  → Comprendre l'environnement MiniGrid
  → Noter les observations clés pour le rapport

ÉTAPE 2: IMPLÉMENTATION Q-LEARNING 
  → Exécuter: python qlearning_agent.py
  → Observer l'apprentissage
  → Analyser les résultats
  → Expérimenter avec les hyperparamètres

ÉTAPE 3: IMPLÉMENTATION DQN 
  → Exécuter: python dqn_agent.py
  → Comparer avec Q-Learning
  → Tester différentes architectures
  → Optimiser les performances

ÉTAPE 4: ÉVALUATION 
  → Intégrer les résultats dans evaluation_analysis.py
  → Générer les visualisations
  → Analyser les graphiques

ÉTAPE 5: RAPPORT 
  → Utiliser le template rapport_tp.md
  → Inclure les graphiques
  → Discuter des résultats
  → Proposer des améliorations


"""
    
    print(workflow)


def show_hyperparameters_guide():
    """Affiche un guide des hyperparamètres."""
    print_section("GUIDE DES HYPERPARAMÈTRES")
    
    guide = """
Q-LEARNING:
  learning_rate (α)     : 0.1 - 0.5
    → Plus élevé = apprentissage plus rapide mais moins stable
  
  gamma (γ)            : 0.95 - 0.99
    → Plus élevé = planification à long terme
  
  epsilon_start        : 1.0
  epsilon_end          : 0.01 - 0.1
  epsilon_decay        : 0.99 - 0.999
    → Contrôle exploration vs exploitation

DQN:
  learning_rate        : 0.0001 - 0.001
    → Plus faible que Q-Learning pour stabilité
  
  batch_size           : 32 - 128
    → Plus grand = plus stable mais plus lent
  
  buffer_capacity      : 10000 - 50000
    → Plus grand = meilleure décorrélation
  
  target_update_freq   : 5 - 20
    → Plus élevé = plus stable mais convergence plus lente
  
  hidden_sizes         : [64, 64] ou [128, 64]
    → Ajuster selon la complexité de l'environnement

CONSEILS:
  - Commencer avec les valeurs par défaut
  - Changer un paramètre à la fois
  - Comparer les courbes d'apprentissage
  - Documenter toutes les expériences
    """
    
    print(guide)


def main():
    """
    Fonction principale.
    """
    print(""" TP APPRENTISSAGE PAR RENFORCEMENT - MINIGRID         
                 Implémentation de Q-Learning et DQN                 
   
    """)
    
    # Menu interactif
    while True:
        print("\nMENU PRINCIPAL:")
        print("  1. Vérifier les dépendances")
        print("  2. Voir la structure du projet")
        print("  3. Voir le workflow recommandé")
        print("  4. Guide des hyperparamètres")
        print("  5. Lancer l'exploration")
        print("  6. Lancer Q-Learning")
        print("  7. Lancer DQN")
        print("  8. Lancer l'évaluation")
        print("  9. Quitter")
        
        choice = input("\nVotre choix (1-9): ").strip()
        
        if choice == '1':
            check_dependencies()
        elif choice == '2':
            show_project_structure()
        elif choice == '3':
            show_workflow()
        elif choice == '4':
            show_hyperparameters_guide()
        elif choice == '5':
            run_exploration()
        elif choice == '6':
            run_qlearning_training()
        elif choice == '7':
            run_dqn_training()
        elif choice == '8':
            run_evaluation()
        elif choice == '9':
            break
        else:
            print("\nChoix invalide. Veuillez choisir entre 1 et 9.")


if __name__ == "__main__":
    main()
