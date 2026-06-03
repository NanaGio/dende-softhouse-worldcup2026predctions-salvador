import os
import pandas as pd
from dende_train import load_and_split_data, train_models

if __name__ == "__main__":
    # Importando CSVs
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Lendo o Train Engineered
    train_data_path = os.path.join(base_dir, 'data', 'train_engineered.csv')
    
    
    test_data_path = os.path.join(base_dir, 'data', 'test_engineered.csv')
    test_original_path = os.path.join(base_dir, 'data', 'test.csv')

    # Estabelecendo o split e treinando
    X_train, X_test, y_train, y_test = load_and_split_data(train_data_path)
    trained_models = train_models(X_train, y_train)

    # Escolhendo o Random Forest
    best_model = trained_models['Random Forest (Tuning)']

    # Lendo os CSVs de Teste
    df_test_features = pd.read_csv(test_data_path)
    df_test_original = pd.read_csv(test_original_path)

    # Retornando as colunas retiradas (apenas por segurança)
    missing_cols = set(X_train.columns) - set(df_test_features.columns)
    for c in missing_cols:
        df_test_features[c] = 0

    df_test_features = df_test_features[X_train.columns]

    # Fazendo a probabilidade
    win_probabilities = best_model.predict_proba(df_test_features)[:, 1]

    # Juntando com os nomes dos times
    results_df = pd.DataFrame({
        'Team': df_test_original['team_name'],
        'Win_Probability_Percentage': win_probabilities * 100
    })

    # Agrupa por país e tira a média das probabilidades
    team_avg_prob = results_df.groupby('Team', as_index=False)['Win_Probability_Percentage'].mean()

    # Ranking ordenado do maior para o menor agora com as médias
    ranking = team_avg_prob.sort_values(by='Win_Probability_Percentage', ascending=False).reset_index(drop=True)

    # Printando ranking (Agora mostrará 10 países ÚNICOS)
    print("\n" + "-"*50)
    print(" ★ RANKING DE PROBABILIDADE MÉDIA DE VENCER A COPA ★")
    print(ranking.head(10)) 

    # Vencedor!!
    champion = ranking.iloc[0]['Team']
    print("\n" + "★" * 20)
    print(f"\n VENCEDOR : ★ {champion.upper()} ★ \n")
    print("\n" + "★" * 20)