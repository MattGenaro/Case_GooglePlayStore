import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Numera_PS/googleplaystore_user_reviews.csv', engine='python')

#parametros estatisticos da base inteira
df.describe()

#localizacao de valores inconsistentes
df.loc[df['Sentiment_Polarity'] < 1]
df.loc[df['Sentiment_Polarity'] > 1]
df.loc[df['Sentiment_Subjectivity'] < 1]
df.loc[df['Sentiment_Subjectivity'] > 1]
#nao ha valores inconsistentes

#localizacao de valores faltantes
pd.isnull(df['Sentiment_Polarity'])
df.loc[pd.isnull(df['Sentiment_Polarity'])]
#26863 valores faltantes
#1a decisao: ignorar
df.dropna(inplace=True)

#valores de parametros estatisticos interessantes sobre a base
df.groupby('App').mean() 
df['Sentiment'].value_counts()

df = df[['Sentiment_Polarity', 'Sentiment', 'Sentiment_Subjectivity']] #fatia o dataframe base em um novo dataframe sem reviews

#reordenamento de atributos para fins de calculo
df0 = df.columns.tolist()
df0 = df0[1:] + df0[:1]
df = df[df0]

#transformacao de variaveis categoricas em variaveis numericas discretas
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sentiment'] = le.fit_transform(df['Sentiment'].astype(str))
#como nao ha valores numericos maiores que modulo de 1, o conjunto de dados ja se encontra normalizado e nao necessita de escalonamento

#define novos conjuntos de dados para serem trabalhados
previsores = df.iloc[:,1:3].values
classe = df.iloc[:,0].values

#matrizes de correlacao de Pearson para os atributos de 'Sentiment_Polarity', 'Sentiment', 'Sentiment_Subjectivity'
np.corrcoef(classe.astype(float), previsores[:,0].astype(float)) #coef 0.17756321
np.corrcoef(classe.astype(float), previsores[:,1].astype(float)) #coef 0.75231962
np.corrcoef(previsores[:,0].astype(float), previsores[:,1].astype(float)) #coef 0.26158728

#divide os conjuntos de dados entre treino e teste
from sklearn.model_selection import train_test_split
previsores_train, previsores_test, classe_train, classe_test = train_test_split(previsores, classe, test_size=0.3, random_state=0)

#modelo de florestas aleatorias
from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier(n_estimators=400, criterion='entropy', random_state=0)
classificador.fit(previsores_train, classe_train)
print(classificador.feature_importances_)
previsoes = classificador.predict(previsores_test) 

#parametros de calculam a acuracia do modelo
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_test, previsoes)
matriz = confusion_matrix(classe_test, previsoes)

#plot do grafico da matriz de confusao
import seaborn as sn
import matplotlib.pyplot as plt
df_cm = pd.DataFrame(matriz, range(3), range(3))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, xticklabels=['Negativo', 'Neutro', 'Positivo'], yticklabels=['Negativo', 'Neutro', 'Positivo'],
           annot=True, annot_kws={"size": 14}, linewidths=.5, fmt="d") # font size
plt.ylabel("Classificação real dos sentimentos")
plt.xlabel("Classificação prevista dos sentimentos")
plt.savefig('cmarfr.png')
plt.show()

#Random Forest com n=10, precisao = 0.9984860628729183
#conclusao:
#modelo com uma unica arvore de decisao de 3 nodos foi melhor que o modelo de Random Forest com 40 arvores
#Random Forest para n=400, precisao = 0.9986641731231632
#superou o modelo de ADD com 3 nodos, mas aumentou a complexidade
