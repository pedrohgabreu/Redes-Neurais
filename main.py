import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data_prep import load_and_preprocess_data
from src.dataset import ClinicalDataset
from src.model import FlexibleMLP
from src.train import train_model
from src.metrics import evaluate_classification
from src.metrics import plot_learning_curve

def main():
    print("1. Preparando dados e Dataloaders...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(task='binary')
    
    train_dataset = ClinicalDataset(X_train, y_train)
    test_dataset = ClinicalDataset(X_test, y_test) # Dataloader de Teste
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("2. Instanciando o Modelo, Loss e Otimizador...")
    model = FlexibleMLP(input_dim=X_train.shape[1], hidden_layers=[64, 32], output_dim=1, dropout_rate=0.2)
    
    # BCEWithLogitsLoss é a melhor função para classificação binária no PyTorch
    criterion = nn.BCEWithLogitsLoss() 
    # Otimizador Adam (padrão de mercado para começar)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\n3. Iniciando o Treinamento...")
    model, history = train_model(model, train_loader, test_loader, criterion, optimizer, epochs=50)
    
    print("\n4. Avaliando o Modelo Final nos dados de Teste...")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_pred_logits = model(X_test_tensor).numpy() # Pega a previsão e converte pra Numpy
        
        # Chama a nossa função de métricas
        metrics = evaluate_classification(y_test, y_pred_logits, task='binary')
        
    print("\n--- Resultados Finais (Classificação Binária) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    # Gera o gráfico exigido na Etapa 4
    plot_learning_curve(history, filename="learning_curve_bin.png")
    
if __name__ == "__main__":
    main()