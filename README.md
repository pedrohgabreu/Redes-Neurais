# Pipeline Fim-a-Fim de Machine Learning com PyTorch e Optuna

Este repositório contém o desenvolvimento de um pipeline completo de Machine Learning utilizando dados clínicos reais para resolver problemas de Classificação Binária, Classificação Multiclasse e Regressão. O projeto foi estruturado de forma modular e extensível, utilizando o ecossistema PyTorch para a construção das redes neurais (MLP) e o Optuna para a otimização automatizada de hiperparâmetros.

## 📋 Estrutura do Projeto

O projeto adota uma arquitetura limpa e modular, separando as responsabilidades de manipulação de dados, definição de modelos, treinamento e avaliação:

trabalho_ml_coracao/
│
├── data/                    # Dados do projeto
│   ├── raw/                 # Dataset original (Heart Disease Cleveland)
│   └── processed/           # Dados após limpeza e padronização
│
├── notebooks/               # Jupyter Notebooks para análise exploratória
│   └── 01_eda_e_limpeza.ipynb
│
├── src/                     # Código-fonte modular (Scripts Python)
│   ├── __init__.py          
│   ├── config.py            # Configurações globais e caminhos de diretórios
│   ├── data_prep.py         # Pipeline de carregamento, limpeza e StandardScaler
│   ├── dataset.py           # Classe nativa PyTorch (ClinicalDataset)
│   ├── model.py             # Arquitetura dinâmica e parametrizável (FlexibleMLP)
│   ├── train.py             # Motor do loop de treinamento (Forward/Backward)
│   ├── metrics.py           # Cálculo de métricas e geração de gráficos de Loss
│   ├── feature_selection.py # Etapa 3: Seleção via Random Forest e SHAP
│   ├── run_tasks.py         # Execução unificada das 3 tarefas (Binária, Multi, Regressão)
│   └── optimize.py          # Etapa 6: Otimização de hiperparâmetros via Optuna
│
├── models/                  # Pesos e artefatos dos modelos treinados (.pth)
├── reports/                 # Relatórios e gráficos exportados
│   └── figures/             # Curvas de aprendizado e gráficos do SHAP
│
├── main.py                  # Script mestre para execução e teste do pipeline
├── requirements.txt         # Dependências do projeto
└── README.md                # Documentação do repositório

---

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos
Certifique-se de ter o Python 3.10+ instalado em sua máquina.

### 2. Clonar o Repositório e Entrar no Diretório
git clone https://github.com/seu-usuario/seu-repositorio.git
cd trabalho_ml_coracao

### 3. Instalar as Dependências
Instale todos os pacotes necessários de forma automatizada via requirements.txt:
pip install -r requirements.txt

### 4. Executar os Scripts do Pipeline

Cada etapa do projeto pode ser disparada diretamente pelo terminal a partir da raiz do repositório:

* Testar o ambiente e carregamento de dados:
  python main.py

* Executar a Etapa 3 (Seleção de Features e Gráfico SHAP):
  python -m src.feature_selection

* Executar as Etapas 4 e 5 (Treinar Classificação Binária, Multiclasse e Regressão):
  python -m src.run_tasks

* Executar a Etapa 6 (Otimização Automatizada com Optuna):
  python -m src.optimize

---

## 🧠 Resumo Teórico e Resultados Obtidos

### Dataset Selecionado (Etapa 1 & 2)
* Origem: Heart Disease Dataset (Cleveland) - UCI Machine Learning Repository.
* Volume: 297 amostras limpas após remoção de valores ausentes (?).
* Abordagem: Utilização do StandardScaler para garantir média 0 e variância 1 em todas as features tabulares, viabilizando a convergência dos gradientes no PyTorch.

### Seleção de Features (Etapa 3)
Utilizou-se o método embutido (Embedded) extraído de um estimador Random Forest. O modelo reduziu a dimensionalidade do problema de 13 para 5 atributos estruturais (ca, cp, thal, thalach, oldpeak), retendo mais de 98% da capacidade preditiva original do sistema e otimizando o tempo de treinamento.

### Arquitetura Flexível (FlexibleMLP)
O grande diferencial técnico deste repositório está na classe FlexibleMLP (src/model.py). Ela foi desenvolvida de forma dinâmica para construir a topologia da rede a partir de parâmetros de entrada, permitindo reusar o mesmo bloco de código para todas as tarefas do roteiro:
* Classificação Binária: Saída linear (1 neurônio) com BCEWithLogitsLoss().
* Classificação Multiclasse: Saída expandida (5 neurônios) com CrossEntropyLoss().
* Regressão: Saída linear contínua (1 neurônio) com MSELoss().

### Resultados Finais Controlados (Classificação Binária)
O modelo base obteve os seguintes resultados utilizando o histórico de treinamento de 50 épocas:
* Acurácia (Accuracy): 0.8167
* Precisão (Precision): 0.7600
* Sensibilidade (Recall): 0.7917
* F1-Score: 0.7755
* ROC-AUC: 0.9352

### Otimização e Regularização (Etapa 6 & 7)
A análise das curvas de aprendizado revelou o surgimento de overfitting na rede padrão após a época 30 (onde a perda de treino caía para 0.2485 e o erro de validação subia para 0.3367). O problema foi mitigado aplicando as diretrizes da otimização automatizada do Optuna. O framework localizou um ponto de operação ótimo que elevou a Acurácia de Teste para 93.33% ao simplificar a rede para apenas 1 camada oculta com 60 neurônios e introduzir uma camada estocástica de Dropout a 44.3%, garantindo a estabilização e a generalização do modelo em dados clínicos não vistos.

---
Desenvolvido para fins acadêmicos como parte dos requisitos da disciplina de Sistemas Inteligentes.
