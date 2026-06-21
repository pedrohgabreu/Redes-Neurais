import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from src.data_prep import load_and_preprocess_data
from src.dataset import ClinicalDataset
from src.model import FlexibleMLP
from src.train import train_model

def plot_comparison(history_overfit, history_reg, filename="comparacao_overfitting.png"):
    save_dir = "reports/figures"
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Gráfico 1: Sem Regularização (Overfitting)
    plt.subplot(1, 2, 1)
    plt.plot(history_overfit['train_loss'], label='Treino', color='blue')
    plt.plot(history_overfit['val_loss'], label='Validação', color='orange')
    plt.title('Modelo Complexo (Sem Regularização)')
    plt.xlabel('Épocas')
    plt.ylabel('Loss (Erro)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Gráfico 2: Com Regularização (Optuna)
    plt.subplot(1, 2, 2)
    plt.plot(history_reg['train_loss'], label='Treino', color='blue')
    plt.plot(history_reg['val_loss'], label='Validação', color='orange')
    plt.title('Modelo Otimizado (Dropout 0.44 + Camada Única)')
    plt.xlabel('Épocas')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n-> Gráfico comparativo salvo com sucesso em: {save_path}")

def main():
    print("--- ETAPA 7: ANÁLISE DE OVERFITTING E REGULARIZAÇÃO ---")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(task='binary')
    
    train_loader = DataLoader(ClinicalDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(ClinicalDataset(X_test, y_test), batch_size=32, shuffle=False)
    
    input_dim = X_train.shape[1]
    epochs = 100 # Mais épocas para forçar o modelo a decorar
    criterion = nn.BCEWithLogitsLoss()
    
    # --- REDE 1: PROPÍCIA A OVERFITTING ---
    print("\n1. Treinando Rede Complexa (Sem regularização, 3 camadas ocultas)...")
    # Forçamos uma arquitetura profunda e sem dropout
    model_overfit = FlexibleMLP(input_dim, hidden_layers=[128, 64, 32], output_dim=1, dropout_rate=0.0)
    optimizer_overfit = optim.Adam(model_overfit.parameters(), lr=0.001)
    
    import sys, io
    original_stdout = sys.stdout
    sys.stdout = io.StringIO() # Oculta os prints do loop
    _, history_overfit = train_model(model_overfit, train_loader, test_loader, criterion, optimizer_overfit, epochs=epochs)
    sys.stdout = original_stdout
    
    # --- REDE 2: OTIMIZADA PELO OPTUNA ---
    print("2. Treinando Rede Otimizada (1 camada, Dropout 0.44, LR ajustada)...")
    model_reg = FlexibleMLP(input_dim, hidden_layers=[60], output_dim=1, dropout_rate=0.44)
    optimizer_reg = optim.Adam(model_reg.parameters(), lr=0.0007)
    
    sys.stdout = io.StringIO()
    _, history_reg = train_model(model_reg, train_loader, test_loader, criterion, optimizer_reg, epochs=epochs)
    sys.stdout = original_stdout
    
    # Gerar gráfico comparativo
    plot_comparison(history_overfit, history_reg)

if __name__ == "__main__":
    main()