# Exercices et Défis Supplémentaires

Ce fichier contient des exercices optionnels pour approfondir votre compréhension de l'apprentissage par renforcement avec MiniGrid.

## 📚 Niveau 1: Expérimentations de base

### Exercice 1.1: Impact du Learning Rate

**Objectif:** Comprendre l'effet du taux d'apprentissage sur la convergence.

**Tâche:**
1. Entraîner des agents Q-Learning avec différents learning rates: [0.01, 0.1, 0.5, 0.9]
2. Comparer les courbes d'apprentissage
3. Identifier le meilleur learning rate pour cet environnement

**Questions:**
- Que se passe-t-il avec un learning rate trop faible?
- Que se passe-t-il avec un learning rate trop élevé?
- Comment choisir le bon learning rate?

**Code de départ:**
```python
learning_rates = [0.01, 0.1, 0.5, 0.9]
results = {}

for lr in learning_rates:
    agent = QLearningAgent(
        action_space_size=env.action_space.n,
        learning_rate=lr
    )
    # Entraîner et stocker les résultats
    results[lr] = agent.train(env, num_episodes=500)

# Comparer les résultats
```

### Exercice 1.2: Stratégies d'Exploration

**Objectif:** Tester différentes stratégies d'exploration.

**Tâche:**
Implémenter et comparer:
1. **Epsilon-greedy** (déjà implémenté)
2. **Epsilon-greedy avec décroissance linéaire** au lieu d'exponentielle
3. **Softmax/Boltzmann exploration**

**Code pour Boltzmann:**
```python
def select_action_boltzmann(self, state, temperature=1.0):
    state_key = self.state_to_key(state)
    q_values = self.q_table[state_key]
    
    # Calculer les probabilités avec softmax
    exp_q = np.exp(q_values / temperature)
    probs = exp_q / np.sum(exp_q)
    
    return np.random.choice(self.action_space_size, p=probs)
```

**Questions:**
- Quelle stratégie converge le plus vite?
- Quelle stratégie est la plus stable?
- Comment le paramètre de température affecte-t-il l'exploration?

### Exercice 1.3: Fonction de Récompense

**Objectif:** Comprendre l'impact du reward shaping.

**Tâche:**
Modifier la récompense pour guider l'agent:

```python
# Exemple: pénalité pour chaque étape
def custom_reward(original_reward, done, steps):
    if done and original_reward > 0:
        return 1.0
    else:
        return -0.01  # Pénalité pour encourager l'efficacité

# Ou: récompense basée sur la distance à l'objectif
def distance_reward(agent_pos, goal_pos):
    distance = np.sqrt((agent_pos[0] - goal_pos[0])**2 + 
                       (agent_pos[1] - goal_pos[1])**2)
    return -distance / 10
```

**Questions:**
- Le reward shaping aide-t-il l'apprentissage?
- Quels sont les risques du reward shaping?
- Comment éviter de biaiser la politique?

## 🚀 Niveau 2: Améliorations algorithmiques

### Exercice 2.1: SARSA

**Objectif:** Implémenter SARSA et le comparer à Q-Learning.

**Différence clé:**
- Q-Learning: utilise `max Q(s',a')` (off-policy)
- SARSA: utilise `Q(s',a')` où `a'` est l'action réellement choisie (on-policy)

**Code à modifier:**
```python
def update_sarsa(self, state, action, reward, next_state, next_action, done):
    state_key = self.state_to_key(state)
    next_state_key = self.state_to_key(next_state)
    
    current_q = self.q_table[state_key][action]
    next_q = 0 if done else self.q_table[next_state_key][next_action]
    
    new_q = current_q + self.learning_rate * (reward + self.gamma * next_q - current_q)
    self.q_table[state_key][action] = new_q
```

**Questions:**
- SARSA est-il plus conservateur que Q-Learning?
- Dans quels cas SARSA est-il préférable?
- Comparez les performances sur MiniGrid.

### Exercice 2.2: Double Q-Learning

**Objectif:** Réduire le biais de surestimation.

**Principe:**
Utiliser deux tables Q et alterner entre elles.

**Code de départ:**
```python
class DoubleQLearningAgent:
    def __init__(self, ...):
        self.q_table_1 = defaultdict(lambda: np.zeros(action_space_size))
        self.q_table_2 = defaultdict(lambda: np.zeros(action_space_size))
    
    def update(self, state, action, reward, next_state, done):
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        
        if np.random.random() < 0.5:
            # Mise à jour Q1
            best_action = np.argmax(self.q_table_1[next_state_key])
            target = reward + self.gamma * self.q_table_2[next_state_key][best_action]
            self.q_table_1[state_key][action] += self.learning_rate * (target - self.q_table_1[state_key][action])
        else:
            # Mise à jour Q2
            best_action = np.argmax(self.q_table_2[next_state_key])
            target = reward + self.gamma * self.q_table_1[next_state_key][best_action]
            self.q_table_2[state_key][action] += self.learning_rate * (target - self.q_table_2[state_key][action])
```

### Exercice 2.3: Prioritized Experience Replay

**Objectif:** Améliorer DQN avec replay buffer priorisé.

**Principe:**
Échantillonner les transitions importantes plus fréquemment.

**Code de départ:**
```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # Degré de priorisation
        self.buffer = []
        self.priorities = []
        self.pos = 0
    
    def push(self, transition):
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_priority
        
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calcul des poids d'importance
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-5
```

## 🎯 Niveau 3: Environnements complexes

### Exercice 3.1: MiniGrid-DoorKey-8x8-v0

**Objectif:** Adapter vos agents à un environnement avec portes et clés.

**Défis:**
- Espace d'états plus grand
- Séquence d'actions requise (trouver clé → ouvrir porte → atteindre objectif)
- Nécessite de la mémoire/planification

**Modifications suggérées:**
```python
env = gym.make('MiniGrid-DoorKey-8x8-v0')

# Augmenter le nombre d'épisodes
num_episodes = 5000

# Ajuster les hyperparamètres
learning_rate = 0.05  # Plus faible pour plus de stabilité
epsilon_decay = 0.999  # Décroissance plus lente
```

**Questions:**
- Vos agents convergent-ils?
- Faut-il modifier la représentation de l'état?
- Comment gérer la dépendance temporelle?

### Exercice 3.2: MiniGrid-FourRooms-v0

**Objectif:** Navigation dans un environnement avec obstacles.

**Défis:**
- Exploration difficile
- Récompense sparse
- Besoin de traverser plusieurs pièces

**Suggestion:**
Implémenter le **curiosity-driven exploration** avec une récompense intrinsèque.

```python
def intrinsic_reward(state_count, state):
    # Récompenser la visite de nouveaux états
    return 1.0 / np.sqrt(state_count[state])
```

### Exercice 3.3: MiniGrid avec obstacles dynamiques

**Objectif:** Gérer un environnement non-stationnaire.

```python
env = gym.make('MiniGrid-Dynamic-Obstacles-8x8-v0')
```

**Questions:**
- L'agent apprend-il une politique robuste?
- Comment adapter l'exploration?
- Faut-il continuer à explorer même après convergence?

## 🏆 Niveau 4: Défis avancés

### Défi 4.1: Meta-Learning

**Objectif:** Entraîner un agent capable de s'adapter rapidement à de nouveaux environnements.

**Approche:**
1. Entraîner sur plusieurs environnements MiniGrid
2. Utiliser les paramètres appris comme initialisation
3. Fine-tuner rapidement sur un nouvel environnement

### Défi 4.2: Hierarchical RL

**Objectif:** Décomposer la tâche en sous-tâches.

**Exemple pour DoorKey:**
- Macro-action 1: Trouver la clé
- Macro-action 2: Aller à la porte
- Macro-action 3: Atteindre l'objectif

### Défi 4.3: Imitation Learning

**Objectif:** Pré-entraîner avec des démonstrations humaines.

**Étapes:**
1. Enregistrer des trajectoires optimales
2. Pré-entraîner avec Behavioral Cloning
3. Fine-tuner avec RL

### Défi 4.4: Multi-Agent RL

**Objectif:** Plusieurs agents coopératifs ou compétitifs.

```python
env = gym.make('MultiGrid-CompetativeRedBlueDoor-v0')
```

## 📊 Exercice d'Analyse

### Analyse comparative complète

**Objectif:** Produire une analyse scientifique rigoureuse.

**Tâches:**
1. Exécuter 10 seeds différents pour chaque algorithme
2. Calculer moyenne et intervalle de confiance
3. Tests statistiques (t-test, Mann-Whitney)
4. Analyse de la variance
5. Graphiques avec barres d'erreur

**Code de départ:**
```python
from scipy import stats

results_qlearning = []
results_dqn = []

for seed in range(10):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Entraîner Q-Learning
    agent_q = QLearningAgent(...)
    rewards_q = agent_q.train(env, ...)
    results_qlearning.append(np.mean(rewards_q[-100:]))
    
    # Entraîner DQN
    agent_dqn = DQNAgent(...)
    rewards_dqn = agent_dqn.train(env, ...)
    results_dqn.append(np.mean(rewards_dqn[-100:]))

# Test statistique
t_stat, p_value = stats.ttest_ind(results_qlearning, results_dqn)
print(f"t-statistic: {t_stat}, p-value: {p_value}")

# Intervalles de confiance à 95%
conf_interval_q = stats.t.interval(0.95, len(results_qlearning)-1,
                                    loc=np.mean(results_qlearning),
                                    scale=stats.sem(results_qlearning))
```

## 🎓 Projet Final Suggéré

### Créer votre propre environnement MiniGrid

**Objectif:** Concevoir et résoudre un environnement personnalisé.

**Étapes:**
1. Définir la tâche (ex: collecte d'objets, puzzle)
2. Créer l'environnement avec MiniGrid
3. Adapter vos agents
4. Analyser les résultats
5. Publier sur GitHub

**Template:**
```python
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv

class CustomEnv(MiniGridEnv):
    def __init__(self, size=8, **kwargs):
        self.size = size
        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            **kwargs
        )
    
    def _gen_grid(self, width, height):
        # Créer la grille
        self.grid = Grid(width, height)
        
        # Ajouter les murs
        self.grid.wall_rect(0, 0, width, height)
        
        # Placer l'agent
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        
        # Placer l'objectif
        self.put_obj(Goal(), width - 2, height - 2)
        
        # Ajouter votre logique personnalisée
        # ...

# Utiliser l'environnement
from gymnasium.envs.registration import register

register(
    id='MiniGrid-Custom-v0',
    entry_point='__main__:CustomEnv'
)

env = gym.make('MiniGrid-Custom-v0')
```

## 📝 Checklist des Exercices

- [ ] Exercice 1.1: Impact du Learning Rate
- [ ] Exercice 1.2: Stratégies d'Exploration
- [ ] Exercice 1.3: Fonction de Récompense
- [ ] Exercice 2.1: SARSA
- [ ] Exercice 2.2: Double Q-Learning
- [ ] Exercice 2.3: Prioritized Experience Replay
- [ ] Exercice 3.1: DoorKey Environment
- [ ] Exercice 3.2: FourRooms Environment
- [ ] Exercice 3.3: Dynamic Obstacles
- [ ] Défi 4.1: Meta-Learning
- [ ] Défi 4.2: Hierarchical RL
- [ ] Défi 4.3: Imitation Learning
- [ ] Défi 4.4: Multi-Agent RL
- [ ] Analyse comparative complète
- [ ] Projet final personnalisé

## 🎯 Critères d'Évaluation

Pour chaque exercice, évaluez-vous selon:

1. **Compréhension** (30%)
   - Comprenez-vous le concept?
   - Pouvez-vous l'expliquer?

2. **Implémentation** (40%)
   - Code fonctionnel?
   - Bonnes pratiques?
   - Commentaires clairs?

3. **Analyse** (30%)
   - Résultats interprétés?
   - Comparaisons pertinentes?
   - Conclusions justifiées?

## 💡 Conseils

1. **Commencez simple:** Validez chaque modification avant de passer à la suivante
2. **Documentez:** Notez tous vos résultats et observations
3. **Visualisez:** Les graphiques aident à comprendre
4. **Comparez:** Toujours avoir une baseline
5. **Itérez:** L'apprentissage par renforcement nécessite de l'expérimentation

Bon courage! 🚀
