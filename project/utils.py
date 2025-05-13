import random
import numpy as np
import torch
import torch.nn as nn

def set_seed(seed: int):
    """
    Setzt den Random Seed für Python, NumPy und Torch (inkl. CUDA) auf `seed` für Reproduzierbarkeit.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Für deterministische CUDNN-Gewährleistung
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """
    Frühzeitiges Stoppen basierend auf Validierungs-Metrik.
    stoppt, wenn sich der beobachtete Wert `metric` nicht innerhalb `patience` Epochen verbessert.
    """
    def __init__(self, patience: int = 5, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.num_bad_epochs = 0

    def step(self, current: float) -> bool:
        """
        Rückgabe True, wenn Training gestoppt werden soll.
        `current` ist z.B. der Validierungs-Loss (kleiner besser).
        """
        if self.best_score is None:
            self.best_score = current
            return False
        # Für Loss: kleinere Werte sind besser
        if current < self.best_score - self.delta:
            self.best_score = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True
        return False
