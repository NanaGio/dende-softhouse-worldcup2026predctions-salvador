import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('../data/train.csv')

#Droping these two, I don't think they are necessary here, only country_code is needed to identify the objects.
df.drop(['team_name','confederation'],axis=1,inplace=True)

#one_hot for classification
df = pd.get_dummies(df, columns=['country_code'], dtype=int)

#Setting target (y) and setting data for learning (X)
X = df.drop(columns=['winner'])
y = df['winner']

#Scaling for context in the numbers
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

df.to_csv('../data/train_clean.csv', index=False)

#test cleaning


