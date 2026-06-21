import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from src.data_prep import load_and_preprocess_data
from src.dataset import ClinicalDataset
from src.model import FlexibleMLP

def objective(trial):
    """
    Função objetivo que o Optuna vai tentar maximizar (neste caso, a Acurácia de Validação).
    """
    # 1. Definindo o Espaço de Busca (Search Space)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_layers = []
    for i in range(n_layers):
        # Sugere entre 16 e 128 neurônios para cada camada
        units = trial.suggest_int(f'n_units_l{i}', 16, 128)
        hidden_layers.append(units)
        
    # Etapa 7: Testando diferentes taxas de regularização (Dropout)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    # Hiperparâmetros de Treinamento
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])

    # 2. Carregando os Dados
    X_train, X_test, y_train, y_test = load_and_preprocess_data(task='binary')
    train_loader = DataLoader(ClinicalDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ClinicalDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # 3. Instanciando o Modelo dinâmico
    input_dim = X_train.shape[1]
    model = FlexibleMLP(input_dim, hidden_layers, output_dim=1, dropout_rate=dropout_rate)
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Seleciona o otimizador com base na sugestão do Optuna
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # 4. Treinamento Rápido para Avaliação (Early Stopping simulado)
    epochs = 30
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # 5. Avaliação final do Trial
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred_logits = model(X_test_tensor).numpy()
        
        # Converte logits para predição binária
        import numpy as np
        y_pred_probs = 1 / (1 + np.exp(-y_pred_logits))
        y_pred_class = (y_pred_probs > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred_class)

    return accuracy

def main():
    print("Iniciando a Otimização de Hiperparâmetros com Optuna...")
    print("Tarefa alvo: Classificação Binária\n")
    
    # Cria o "estudo" focando em MAXIMIZAR a acurácia
    study = optuna.create_study(direction='maximize', study_name="Otimizacao_MLP_Coracao")
    
    # Executa 30 testes (trials) diferentes
    study.optimize(objective, n_trials=30)
    
    print("\n" + "="*40)
    print(" OTIMIZAÇÃO CONCLUÍDA")
    print("="*40)
    
    trial = study.best_trial
    print(f"Melhor Acurácia encontrada: {trial.value:.4f}")
    print("\nMelhores Hiperparâmetros:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()