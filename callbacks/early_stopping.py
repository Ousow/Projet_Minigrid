import copy

class EarlyStopping:
    def __init__(self, monitor='eval_reward', mode='max', patience=50, 
                 min_delta=0.001, restore_best_weights=True, verbose=True):
        
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_score = -float('inf') if mode == 'max' else float('inf')
        self.best_epoch = 0
        self.best_weights = None
        self.counter = 0
        self.stop_training = False
    
    def step(self, metrics, model=None, epoch=None):
        current_score = metrics.get(self.monitor, 0)
        
        if self.mode == 'max':
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = current_score
            self.best_epoch = epoch if epoch is not None else 0
            self.counter = 0
            
            if self.restore_best_weights and model is not None:
                self.best_weights = copy.deepcopy(model.state_dict())
            
            if self.verbose:
                print(f"\n✨ Meilleur score: {self.best_score:.4f} (épisode {self.best_epoch})")
        else:
            self.counter += 1
            if self.verbose and self.counter % 10 == 0:
                print(f"⚠️ Patience: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.stop_training = True
                
                if self.verbose:
                    print(f"\n🛑 EARLY STOPPING à l'épisode {epoch}")
                    print(f"   Meilleur score: {self.best_score:.4f}")
                    print(f"   Meilleur épisode: {self.best_epoch}")
                
                if self.restore_best_weights and self.best_weights is not None and model is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print(f"   ✅ Meilleurs poids restaurés")
        
        return self.stop_training