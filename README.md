# TP Apprentissage par Renforcement avec MiniGrid

Implémentation complète d'agents d'apprentissage par renforcement (Q-Learning et DQN) pour l'environnement MiniGrid.

## Table des matières

- [Description](#description)
- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Utilisation](#utilisation)
- [Algorithmes implémentés](#algorithmes-implémentés)
- [FAQ](#faq)

## Description

Ce projet implémente deux algorithmes d'apprentissage par renforcement:
- **Q-Learning tabulaire**: Approche classique avec table Q
- **Deep Q-Network (DQN)**: Approche moderne avec réseau de neurones

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
├── qlearning_agent.py             # Partie 2A: Q-Learning
├── dqn_agent.py                   # Partie 2B: DQN
├── evaluation_&_analyse.py         # Partie 3: Évaluation
│
├── rapport_tp.md                    # Template de rapport
│
├── results/                         # Résultats générés (créé auto)
│   ├── training_curves.png
│   ├── evaluation_comparison.png
│   ├── reward_distribution.png
│   └── performance_report.txt
│
├── qlearning_agent.pkl             # Agent Q-Learning sauvegardé
└── dqn_agent.pth                   # Agent DQN sauvegardé
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


#### Partie 2A: Entraînement Q-Learning

```bash
python qlearning_agent.py
```

Entraîne un agent Q-Learning avec:
- 1000 épisodes d'entraînement
- Évaluation sur 100 épisodes
- Sauvegarde automatique


#### Partie 2B: Entraînement DQN

```bash
python dqn_agent.py
```

Entraîne un agent DQN avec:
- Réseau de neurones à 2 couches cachées
- Replay buffer et target network
- Sauvegarde automatique


#### Partie 3: Évaluation et analyse

```bash
python evaluation_&_analyse.py
```

Génère:
- Courbes d'apprentissage
- Comparaison des performances
- Rapport détai

## Algorithmes implémentés

### Q-Learning

**Caractéristiques:**
- Algorithme de différence temporelle
- Table Q pour stocker les valeurs état-action
- Politique epsilon-greedy
- Mise à jour: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

**Hyperparamètres par défaut:**
```python
learning_rate = 0.1
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
```

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

### Personnaliser les visualisations

Éditez `evaluation_&_analyse.py` pour modifier:
- Les couleurs des graphiques
- Les métriques affichées
- Le format du rapport

##  Compléter le rapport

1. Exécutez tous les scripts pour générer les résultats
2. Ouvrez `rapport_tp.md`
3. Complétez chaque section avec:
   - Vos observations
   - Les valeurs obtenues
   - Votre analyse critique
4. Ajoutez les images dans le dossier `results/`
5. Exportez en PDF si nécessaire

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

### Comment sauvegarder/charger un agent?

Les agents sont sauvegardés automatiquement. Pour charger:

```python
# Q-Learning
agent = QLearningAgent(...)
agent.load('qlearning_agent.pkl')

# DQN
agent = DQNAgent(...)
agent.load('dqn_agent.pth')
```

##  Ressources supplémentaires

- [Documentation MiniGrid](https://minigrid.farama.org/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Sutton & Barto - Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html)
- [DQN Paper (DeepMind)](https://www.nature.com/articles/nature14236)



---

