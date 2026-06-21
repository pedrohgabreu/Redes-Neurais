import torch
from torch.utils.data import Dataset

class ClinicalDataset(Dataset):
    def __init__(self, X, y):
        """
        Transforma os arrays do Scikit-Learn/Pandas em Tensores do PyTorch.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        
        # Define o tipo do tensor de acordo com a tarefa (CrossEntropy exige long)
        if y.dtype in ['int', 'int32', 'int64'] and len(set(y)) > 2:
            self.y = torch.tensor(y, dtype=torch.long) 
        else:
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # Adiciona dimensão para compatibilidade

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]