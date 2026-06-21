import numpy as np
import matplotlib.pyplot as plt
import os
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader

# Importando os nossos módulos
from src.data_prep import load_and_preprocess_data
from src.dataset import ClinicalDataset
from src.model import FlexibleMLP
from src.train import train_model
from src.metrics import evaluate_classification

def main():
    print("--- ETAPA 3: SELEÇÃO DE FEATURES ---")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(task='binary')
    
    # Nomes das colunas do dataset de Cleveland
    feature_names = np.array([
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ])
    
    print("\n1. Extraindo importância com Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nRanking de Importância:")
    for i in range(len(feature_names)):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
    # Vamos selecionar as TOP 5 features mais importantes
    top_n = 5
    selected_indices = indices[:top_n]
    selected_features = feature_names[selected_indices]
    print(f"\n-> Features Selecionadas (Top {top_n}): {list(selected_features)}")
    
    print("\n2. Gerando Interpretabilidade com SHAP...")
    save_dir = "reports/figures"
    os.makedirs(save_dir, exist_ok=True)
    
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    
    # Dependendo da versão do SHAP, Random Forest retorna lista de arrays (um por classe)
    shap_vals_positiva = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_positiva, X_test, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot - Impacto das Variáveis Clínicas", y=1.05)
    plt.savefig(os.path.join(save_dir, "shap_summary.png"), bbox_inches='tight')
    plt.close()
    print(f"Gráfico SHAP salvo em: {save_dir}/shap_summary.png")
    
    print("\n3. Comparando MLP: Todas as Features vs Selecionadas")
    
    # Filtrando as bases de dados apenas para as 5 melhores variáveis
    X_train_sel = X_train[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]
    
    # Função auxiliar para não repetirmos código de treinamento aqui
    def train_and_eval(X_tr, y_tr, X_te, y_te, input_dim):
        train_loader = DataLoader(ClinicalDataset(X_tr, y_tr), batch_size=32, shuffle=True)
        test_loader  = DataLoader(ClinicalDataset(X_te, y_te), batch_size=32, shuffle=False)
        
        model = FlexibleMLP(input_dim, hidden_layers=[64, 32], output_dim=1, dropout_rate=0.2)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Treina silenciosamente (50 épocas)
        import sys, io
        original_stdout = sys.stdout
        sys.stdout = io.StringIO() # Suprime prints do loop
        model, _ = train_model(model, train_loader, test_loader, criterion, optimizer, epochs=50)
        sys.stdout = original_stdout # Restaura prints
        
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.tensor(X_te, dtype=torch.float32)).numpy()
            metrics = evaluate_classification(y_te, y_pred, task='binary')
        return metrics

    metrics_all = train_and_eval(X_train, y_train, X_test, y_test, input_dim=13)
    metrics_sel = train_and_eval(X_train_sel, y_train, X_test_sel, y_test, input_dim=top_n)
    
    print("\n--- Resultados Finais (Acurácia, F1-Score etc) ---")
    print(f"{'Métrica':<15} | {'13 Features':<15} | {'5 Features':<15}")
    print("-" * 50)
    for k in metrics_all.keys():
        print(f"{k:<15} | {metrics_all[k]:<15.4f} | {metrics_sel[k]:<15.4f}")

if __name__ == "__main__":
    main()