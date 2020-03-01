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

#modelo de arvore de decisao
from sklearn.tree import DecisionTreeClassifier, export
classificador = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0, criterion='entropy')
classificador.fit(previsores_train, classe_train)
print(classificador.feature_importances_) #[0.55087359 0.21414547 0.05557204 0.02206498 0.08550559 0.07183833] para n nodos
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
plt.savefig('cmaddr.png')
plt.show()

#gerador de grafico para Árvore de Decisão
export.export_graphviz(classificador,
                       out_file = 'arvore_reviews3n.dot',
                       feature_names = ['Sentiment_Subjectivity', 'Sentiment_Polarity'],
                       class_names = ['Negative', 'Neutral', 'Positive'],
                       filled = True,
                       leaves_parallel=True)

#foi feita uma primeira análise com o número de nodos livre e obteve-se precisao de 0.9986641731231632
#observou-se que houve um overfitting dos dados devido ao numero de nodos
#decisao: diminuir o numero de nodos para k=3

#precisao = 0.998797755810847
#conclusao:
#em uma arvore de decisao com 3 nós, houve um ganho de 0.001% de precisao, o que mostra que o atributo de polaridade 
#é um fator decisivo para se prever o sentimento










