import os
import pandas as pd
from dende_train import load_and_split_data, train_models

if __name__ == "__main__":
    #Importando CSVs
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_data_path = os.path.join(base_dir, 'data', 'train_engineered.csv')
    test_clean_path = os.path.join(base_dir, 'data', 'test_clean.csv')
    test_original_path = os.path.join(base_dir, 'data', 'test.csv')

    #Estabelecendo o split
    X_train, X_test, y_train, y_test = load_and_split_data(train_data_path)
    trained_models = train_models(X_train, y_train)

    #Escolhendo qual modelo usar, julgamos um modelo de árvore de decisão como o mais preciso para o objetivo final,
    #já que a lógica de négocio aqui, é separar vitórias e contabilizar o mais provável de vencer.
    best_model = trained_models['Random Forest (Tuning)']

    # Lendo os CSVs
    df_test_features = pd.read_csv(test_clean_path)

    # Aqui nós carregamos o CSV original, apenas para resgatar a coluna de nome dos times, que foi descartada no treino do modelo.
    df_test_original = pd.read_csv(test_original_path)

    #Retornando as coluanas retiradas
    missing_cols = set(X_train.columns) - set(df_test_features.columns)
    for c in missing_cols:
        df_test_features[c] = 0


    df_test_features = df_test_features[X_train.columns]

    # Fazendo a probabilidade
    win_probabilities = best_model.predict_proba(df_test_features)[:, 1]

    # Definimos os nomes junto com a probabilidade de vencer, já que o dataset original foi carregado denovo, e está separado
    results_df = pd.DataFrame({
        'Team': df_test_original['team_name'],
        'Win_Probability_Percentage': win_probabilities * 100
    })

    # Ranking maior para o menor
    ranking = results_df.sort_values(by='Win_Probability_Percentage', ascending=False).reset_index(drop=True)

    # Printando ranking
    print("\n" + "-"*50)
    print(" ★ RANKING DE PROBABILIDADE DE VENCER A COPA ★")
    print(ranking.head(10)) # Mostra o Top 10

    # Vencedor!!
    champion = ranking.iloc[0]['Team']
    print("\n" + "★" * 20)
    print(f"\n VENCEDOR : ★ {champion.upper()} ★ \n")
    print("\n" + "★" * 20)