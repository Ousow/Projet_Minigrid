import os
import json
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path

class TrainingCallback:
    def __init__(self, log_dir="runs", model_dir="models", experiment_name=None):
        self.log_dir = Path(log_dir)
        self.model_dir = Path(model_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"dqn_{timestamp}"
        self.experiment_name = experiment_name
        
        self.writer = SummaryWriter(log_dir=str(self.log_dir / experiment_name))
        
        self.metrics = {
            'rewards': [],
            'losses': [],
            'epsilons': [],
            'eval_rewards': [],
            'eval_success_rates': []
        }
        
        print(f"\n🎯 TensorBoard: {self.log_dir / experiment_name}")
        print(f"   tensorboard --logdir={self.log_dir}\n")
    
    def on_episode_end(self, episode, reward, epsilon=None, loss=None):
        self.metrics['rewards'].append(reward)
        if epsilon is not None:
            self.metrics['epsilons'].append(epsilon)
        if loss is not None:
            self.metrics['losses'].append(loss)
        
        self.writer.add_scalar('Train/Reward', reward, episode)
        if epsilon is not None:
            self.writer.add_scalar('Train/Epsilon', epsilon, episode)
        if loss is not None:
            self.writer.add_scalar('Train/Loss', loss, episode)
        
        if len(self.metrics['rewards']) >= 100:
            avg_reward = np.mean(self.metrics['rewards'][-100:])
            self.writer.add_scalar('Train/Avg_Reward_100', avg_reward, episode)
    
    def on_evaluation_end(self, episode, mean_reward, success_rate):
        self.metrics['eval_rewards'].append(mean_reward)
        self.metrics['eval_success_rates'].append(success_rate)
        
        self.writer.add_scalar('Eval/Mean_Reward', mean_reward, episode)
        self.writer.add_scalar('Eval/Success_Rate', success_rate, episode)
    
    def close(self):
        self.writer.close()
        self._save_metrics()
    
    def _save_metrics(self):
        path = self.log_dir / self.experiment_name / 'metrics.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        
        json_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, list):
                json_metrics[key] = [float(v) if hasattr(v, 'item') else v for v in value]
        
        with open(path, 'w') as f:
            json.dump(json_metrics, f, indent=2)