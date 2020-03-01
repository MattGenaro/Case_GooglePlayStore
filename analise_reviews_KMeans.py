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

#novo conjunto de dados para serem trbalhados
x = df.iloc[:,[1,2]].values

#para o modelo de K-Means, e necessario fazer o escalonamento dos dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

#K-Means bib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#escolhendo o melhor numero de clusters via valores e grafico
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.xlabel('Numero de clusters')
plt.ylabel('WCSS')


#modelo de K-Means com numero de clusters definido, n=3
kmeans = KMeans(n_clusters = 3, random_state = 0)
previsoes = kmeans.fit_predict(x)
#visualizacao grafica da clusterizacao n=3
plt.figure(figsize=(10,7))
plt.scatter(x[previsoes == 0, 0], x[previsoes == 0, 1], s = 50, c = 'green', label = 'Cluester 1')
plt.scatter(x[previsoes == 1, 0], x[previsoes == 1, 1], s = 50, c = 'blue', label = 'Cluester 2')
plt.scatter(x[previsoes == 2, 0], x[previsoes == 2, 1], s = 50, c = 'red', label = 'Cluester 3')
plt.xlabel('Sentiment_Subjectivity')
plt.ylabel('Sentiment_Polarity')
#plt.savefig('kmeans3.png')
plt.legend()


#modelo de K-Means com numero de clusters definido, n=5
kmeans = KMeans(n_clusters = 5, random_state = 0)
previsoes = kmeans.fit_predict(x)
#visualizacao grafica da clusterizacao n=5
plt.figure(figsize=(10,7))
plt.scatter(x[previsoes == 0, 0], x[previsoes == 0, 1], s = 50, c = 'red', label = 'Cluester 1')
plt.scatter(x[previsoes == 1, 0], x[previsoes == 1, 1], s = 50, c = 'blue', label = 'Cluester 2')
plt.scatter(x[previsoes == 2, 0], x[previsoes == 2, 1], s = 50, c = 'green', label = 'Cluester 3')
plt.scatter(x[previsoes == 3, 0], x[previsoes == 3, 1], s = 50, c = 'orange', label = 'Cluester 4')
plt.scatter(x[previsoes == 4, 0], x[previsoes == 4, 1], s = 50, c = 'pink', label = 'Cluester 5')
plt.xlabel('Sentiment_Subjectivity')
plt.ylabel('Sentiment_Polarity')
#plt.savefig('kmeans5.png')
plt.legend()

#redefinindo o dataframe para incluir app e translated review novamente
df = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Numera_PS/googleplaystore_user_reviews.csv', engine='python')
df.dropna(inplace=True)

#junta a base de dados com os resultados obtidos da clusterizacao
nova_class = np.column_stack((df, previsoes))
nova_class = nova_class[nova_class[:,4].argsort()]
nova_class = pd.DataFrame(data=nova_class)

