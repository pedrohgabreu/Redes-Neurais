import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data_prep import load_and_preprocess_data
from src.dataset import ClinicalDataset
from src.model import FlexibleMLP
from src.train import train_model
from src.metrics import evaluate_classification, evaluate_regression

def run_experiment(task_name, input_dim, output_dim, criterion, hidden_layers=[64, 32], epochs=50):
    print(f"\n{'='*40}")
    print(f" INICIANDO TAREFA: {task_name.upper()}")
    print(f"{'='*40}")
    
    # 1. Carrega dados específicos para a tarefa
    X_train, X_test, y_train, y_test = load_and_preprocess_data(task=task_name)
    
    # 2. Prepara Dataloaders
    train_loader = DataLoader(ClinicalDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(ClinicalDataset(X_test, y_test), batch_size=32, shuffle=False)
    
    # 3. Instancia o Modelo e Otimizador
    model = FlexibleMLP(input_dim, hidden_layers, output_dim, dropout_rate=0.2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Treina o Modelo
    model, history = train_model(model, train_loader, test_loader, criterion, optimizer, epochs=epochs)
    
    # 5. Avaliação
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred_raw = model(X_test_tensor).numpy()
        
        if task_name == 'binary':
            metrics = evaluate_classification(y_test, y_pred_raw, task='binary')
            
        elif task_name == 'multiclass':
            # Para multiclasse, aplicamos argmax para pegar a classe com maior probabilidade
            import numpy as np
            y_pred_class = np.argmax(y_pred_raw, axis=1)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred_class),
                'Precision Macro': precision_score(y_test, y_pred_class, average='macro', zero_division=0),
                'Recall Macro': recall_score(y_test, y_pred_class, average='macro', zero_division=0),
                'F1 Macro': f1_score(y_test, y_pred_class, average='macro', zero_division=0)
            }
            
        elif task_name == 'regression':
            metrics = evaluate_regression(y_test, y_pred_raw)
            
    print(f"\n--- Resultados Finais ({task_name}) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

def main():
    # As features de entrada (13 anatômicas/clínicas)
    input_features = 13 
    
    # --- ETAPA 4.1: CLASSIFICAÇÃO BINÁRIA ---
    run_experiment(
        task_name='binary',
        input_dim=input_features,
        output_dim=1, # Uma saída predizendo probabilidade (0 ou 1)
        criterion=nn.BCEWithLogitsLoss()
    )
    
    # --- ETAPA 4.2: CLASSIFICAÇÃO MULTICLASSE ---
    run_experiment(
        task_name='multiclass',
        input_dim=input_features,
        output_dim=5, # 5 classes de severidade da doença (0 a 4)
        criterion=nn.CrossEntropyLoss() # Melhor Loss para multiclasse no PyTorch
    )
    
    # --- ETAPA 5: REGRESSÃO ---
    # Na regressão, vamos prever o colesterol (1 variável), logo sobraram 12 features de entrada
    run_experiment(
        task_name='regression',
        input_dim=12, 
        output_dim=1, # Uma saída prevendo um valor contínuo
        criterion=nn.MSELoss() # Erro Quadrático Médio
    )

if __name__ == "__main__":
    main()