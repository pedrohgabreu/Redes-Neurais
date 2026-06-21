import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import RAW_DATA_PATH

def load_and_preprocess_data(task='binary'):
    """
    Baixa os dados (se necessário), limpa e separa em treino/teste.
    task: 'binary', 'multiclass', ou 'regression'
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
    
    # Carrega os dados tratando '?' como nulo
    df = pd.read_csv(url, names=columns, na_values="?")
    
    # Etapa 2: Pré-processamento - Tratamento de valores ausentes (Drop)
    df = df.dropna()
    
    # Salva o raw localmente (opcional, apenas para ter registro)
    df.to_csv(RAW_DATA_PATH, index=False)

    # Separa features (X) e target (y) dependendo da tarefa
    if task == 'regression':
        # Exemplo: Prever o colesterol (chol) usando o resto
        y = df['chol'].values
        X = df.drop(columns=['chol', 'num']).values
    else:
        # Classificação Binária ou Multiclasse
        X = df.drop(columns=['num']).values
        y = df['num'].values
        
        if task == 'binary':
            # Binarizando: 0 = saudável, > 0 = doente
            y = (y > 0).astype(int)
            
    # Separação Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalização (Etapa 2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test