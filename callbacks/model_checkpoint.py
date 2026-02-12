import torch
from pathlib import Path
import json
from datetime import datetime

class ModelCheckpoint:
    def __init__(self, filepath='models/best_model.pth', monitor='eval_reward', 
                 mode='max', save_best_only=True, verbose=True):
        
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        self.best_score = -float('inf') if mode == 'max' else float('inf')
        self.best_epoch = 0
    
    def step(self, model, epoch, metrics):
        current_score = metrics.get(self.monitor, 0)
        
        if self.mode == 'max':
            improved = current_score > self.best_score
        else:
            improved = current_score < self.best_score
        
        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_score': self.best_score,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(checkpoint, self.filepath)
            
            # Sauvegarde JSON
            metrics_path = self.filepath.with_suffix('.json')
            with open(metrics_path, 'w') as f:
                json.dump({
                    'epoch': epoch,
                    'best_score': float(self.best_score),
                    'metrics': {k: float(v) if hasattr(v, 'item') else v 
                              for k, v in metrics.items()},
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            if self.verbose:
                print(f"\n Meilleur modèle sauvegardé!")
                print(f"   Épisode: {epoch}")
                print(f"   {self.monitor}: {self.best_score:.4f}")
                print(f"   Chemin: {self.filepath}")
            
            return True
        return False