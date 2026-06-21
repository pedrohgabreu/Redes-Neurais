import os

# Caminhos base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'heart_disease.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'heart_processed.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Configurações globais
RANDOM_SEED = 42

# Criar diretórios caso não existam
os.makedirs(os.path.join(DATA_DIR, 'raw'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'processed'), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)