import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def load_and_split_data(filepath, target_col='winner', test_size=0.2, random_state=42):
    """
    Carrega o dataset e realiza a divisão de treino e teste.
    Garante que o modelo será avaliado com dados que nunca viu (80% treino / 20% teste).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        
    print(f"Carregando dados de {filepath}...")
    df = pd.read_csv(filepath)
    
    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada no dataset.")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"Separando a base de dados (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    print(f"Tamanho do conjunto de treino: {X_train.shape[0]} amostras")
    print(f"Tamanho do conjunto de teste: {X_test.shape[0]} amostras")
    
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train, random_state=42):
    """
    Treina múltiplos modelos de classificação nos dados de treino.
    Retorna um dicionário com os modelos treinados.
    """
    models = {
        'Random Forest (Default)': RandomForestClassifier(random_state=random_state),
        'Random Forest (Tuning)': RandomForestClassifier(n_estimators=200, max_depth=5, random_state=random_state),
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000, C=0.5)
    }
    
    trained_models = {}
    print("\nIniciando o treinamento dos modelos...")
    for name, model in models.items():
        print(f"Treinando {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name} treinado com sucesso.")
        
    return trained_models

def extract_predictions_and_probabilities(models, X_test):
    """
    Realiza as previsões e extrai as probabilidades de vitória usando .predict_proba()
    para cada modelo treinado.
    """
    results = {}
    print("\nExtraindo predições e probabilidades...")
    for name, model in models.items():
        print(f"Processando {name}...")
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        results[name] = {
            'predictions': predictions,
            'probabilities': probabilities,
            'classes': model.classes_
        }
        print(f"Probabilidades extraídas para {name}. Classes: {model.classes_}")
        
    return results

if __name__ == "__main__":
    # Caminho padrão para os dados processados (ajustado para a estrutura do projeto)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'train_engineered.csv')
    
    # 1. Divisão Treino/Teste
    X_train, X_test, y_train, y_test = load_and_split_data(data_path)
    
    # 2. Treino do Algoritmo
    trained_models = train_models(X_train, y_train)
    
    # 3. Extração de Predições e Probabilidades
    prediction_results = extract_predictions_and_probabilities(trained_models, X_test)
    
    # 4. Avaliação com Acurácia e F1-Measure
    print("\nResultados da Avaliação:")
    for name, results in prediction_results.items():
        acc = accuracy_score(y_test, results['predictions'])
        f1 = f1_score(y_test, results['predictions'], average='weighted')
        print(f"{name} - Acurácia: {acc:.4f} | F1-Score: {f1:.4f}")
