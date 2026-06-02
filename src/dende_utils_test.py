import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
df2 = pd.read_csv('../data/test.csv')

#Droping these two, I don't think they are necessary here, only country_code is needed to identify the objects.
df2.drop(['team_name','confederation'],axis=1,inplace=True)

#one_hot for classification
df2 = pd.get_dummies(df2, columns=['country_code'], dtype=int)

#Setting target (y) and setting data for learning (X)
X = df2

#Scaling for context in the numbers
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

df2.to_csv('../data/test_clean.csv', index=False)



