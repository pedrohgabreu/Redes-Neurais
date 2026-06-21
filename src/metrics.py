import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def plot_learning_curve(history, filename="learning_curve_bin.png"):
    """
    Plota a curva de loss por época e salva na pasta reports/figures/
    """
    save_dir = "reports/figures"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_loss'], label='Loss de Treino', marker='o', markersize=4)
    plt.plot(history['val_loss'], label='Loss de Validação', marker='o', markersize=4)
    plt.title('Curva de Aprendizado - MLP')
    plt.xlabel('Épocas')
    plt.ylabel('Loss (Erro)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo com sucesso em: {save_path}")

def evaluate_classification(y_true, y_pred_logits, task='binary'):
    """
    Calcula as métricas exigidas para Classificação Binária ou Multiclasse.
    """
    if task == 'binary':
        # Aplica Sigmoid para converter logits em probabilidades (0 a 1)
        y_pred_probs = 1 / (1 + np.exp(-y_pred_logits)) 
        y_pred_class = (y_pred_probs > 0.5).astype(int)
        
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred_class),
            'Precision': precision_score(y_true, y_pred_class, zero_division=0),
            'Recall': recall_score(y_true, y_pred_class, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred_class, zero_division=0),
            'ROC-AUC': roc_auc_score(y_true, y_pred_probs)
        }
    else:
        # Lógica para Multiclasse (faremos na etapa multiclasse usando Softmax/Argmax)
        pass 
        
    return metrics

def evaluate_regression(y_true, y_pred):
    """
    Calcula as métricas exigidas para Regressão.
    """
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred)
    }
    return metrics