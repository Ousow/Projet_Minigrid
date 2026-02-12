# TP Apprentissage par Renforcement avec MiniGrid

Implémentation complète d'agents d'apprentissage par renforcement (DQN et DQN_CNN) pour l'environnement MiniGrid.

## Table des matières

- [Description](#description)
- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Utilisation](#utilisation)
- [Algorithmes implémentés](#algorithmes-implémentés)
- [FAQ](#faq)

## Description

Ce projet implémente deux algorithmes d'apprentissage par renforcement:
- **Deep Q-Network (DQN)**: Approche moderne avec réseau de neurones
- **Deep Q-Network (DQN)**: Approche avec des couches convolutives et utilisation de callbacks

L'objectif est de comparer leurs performances sur l'environnement MiniGrid-Empty-8x8-v0.

## Installation

### Prérequis

- Python 3.8+
- pip

### Installation des dépendances

```bash
pip install -r requirements.txt
```

Ou manuellement:

```bash
pip install gymnasium minigrid numpy matplotlib torch tqdm
```

### Vérification de l'installation

```bash
python main.py
# Choisir l'option 1 pour vérifier les dépendances
```

##  Structure du projet

```
tp_minigrid/
│
├── README.md                        # Ce fichier
├── requirements.txt                 # Dépendances Python
├── main.py                          # Script principal avec menu
│
├── exploration_minigrid.py        # Partie 1: Exploration
├── dqn_agent.py                   # Partie 1B: DQN
├── dqn_cnn.py                     # Partie 2B: DQN_CNN
├── visualiser_resultats.py         # Partie 3: Évaluation
│
├── rapport_tp.md                    # Template de rapport
│
├── results/                         # Résultats générés (créé auto)
│   ├── courbes_apprentissage_dqn.png
│   ├── courbes_apprentissage_dqn.png
│   ├── dqn_loss.png
|   ├── performance_report.txt
|   ├── training_progress.png
│   └── resultats_dqn_minigrid_final.png
│
├── benchmark_results
|   ├── comparaison_dqn.png      
|     
└── dqn_agent.pth                  
```

##  Utilisation

### Option 1: Menu interactif (recommandé pour débutants)

```bash
python main.py
```

Naviguez dans le menu pour:
1. Vérifier les dépendances
2. Voir le workflow recommandé
3. Lancer chaque partie du TP

### Option 2: Exécution directe (pour utilisateurs avancés)

#### Partie 1: Exploration de MiniGrid

```bash
python exploration_minigrid.py
```

Cette étape vous permet de:
- Comprendre l'espace d'observations
- Découvrir les actions disponibles
- Analyser le système de récompenses


#### Partie 2A: Entraînement Deep Q-Learning

```bash
python dqn_agent.py
```

Entraîne un agent DQN avec:
- Réseau de neurones à 2 couches cachées
- Replay buffer et target network
- 1000 épisodes d'entraînement
- Évaluation sur 100 épisodes
- Sauvegarde automatique


#### Partie 2B: Entraînement DQN+CNN

```bash
python dqn_cnn.py
```

Entraîne un agent DQN avec:
- Intégration de couches de convolution
- Utilisation des callbacks
- Réseau de neurones à 2 couches cachées
- Replay buffer et target network


#### Partie 3: Visualisation des résultats

```bash
python visualiser_resultats.py
```


## Algorithmes implémentés

### Deep Q-Network (DQN)

**Caractéristiques:**
- Approximation de Q par réseau de neurones
- Experience replay buffer
- Target network pour stabilité
- Gradient clipping

**Architecture par défaut:**
```python
Input Layer: 147 neurones (grille 7x7x3)
Hidden Layer 1: 128 neurones + ReLU
Hidden Layer 2: 64 neurones + ReLU
Output Layer: 7 neurones (actions)
```

**Hyperparamètres par défaut:**
```python
learning_rate = 0.001
batch_size = 64
buffer_capacity = 10000
target_update_freq = 10
```

### DQN_CNN

**Caractéristiques:**
- Approximation par Réseau de Neurones  
- Experience Replay 
- Target Network 
- Politique epsilon-greedy 
- Mise à jour par Descente de Gradient 
    Loss = MSE(r + \gamma \max_{a'} Q_{target}(s', a') - Q_{policy}(s, a))$$

**Hyperparamètres utilisés :**
```python
learning_rate  = 0.0005   
gamma          = 0.99     
epsilon_start  = 1.0      
epsilon_end    = 0.05     
epsilon_decay  = 0.997    
batch_size     = 32       
target_update  = 500 
```

### Conversion en PDF

```bash
# Avec pandoc
pandoc rapport_tp.md -o rapport_tp.pdf

# Avec markdown-pdf (VS Code extension)
# Ou copier-coller dans Google Docs/Word
```

##  FAQ

### L'entraînement est trop long, que faire?

- Réduisez `num_episodes` (ex: 500 au lieu de 1000)
- Utilisez un GPU pour DQN
- Réduisez `max_steps` par épisode

### L'agent ne converge pas

- Vérifiez les hyperparamètres
- Augmentez le learning rate
- Augmentez epsilon_decay pour plus d'exploration
- Vérifiez que l'environnement est correct


### Les graphiques ne s'affichent pas

- Vérifiez que matplotlib est installé
- Les graphiques sont sauvegardés dans `results/`
- Ouvrez les fichiers PNG directement

### Erreur "module not found"

```bash
# Réinstallez les dépendances
pip install --upgrade gymnasium minigrid numpy matplotlib torch tqdm
```

```

##  Ressources supplémentaires

- [Documentation MiniGrid](https://minigrid.farama.org/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Sutton & Barto - Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html)
- [DQN Paper (DeepMind)](https://www.nature.com/articles/nature14236)



---

