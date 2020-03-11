import pandas as pd
import numpy as np
df = pd.read_csv('C:/googleplaystorecopy.csv', engine='python')
#foi feito um preprocessamento direto na base de dados usando Excel antes de realizar o preprocessamento com Python

#parametros estatisticos da base inteira
df.describe()

#localizacao de valores inconsistentes em Rating
df.loc[df['Rating'] < 0]
df.loc[df['Rating'] > 5]
#localizado o total de '1' linha com dado insconsistente
#decisao: desconsiderar linha
df.drop(df[df.Rating > 5].index, inplace=True)

#localizacao de valores faltantes
pd.isnull(df['Rating'])
df.loc[pd.isnull(df['Rating'])]
#1474 valores faltantes
#Imputer: media

#problemas ao se preencher os valores faltantes da coluna Rating e Size com import SimpleImputer
#solucao, usar a funcao fillna antes de dar continuidade ao preprocessamento
df['Rating'].fillna(method='bfill', inplace=True)
df['Size'].fillna(method='bfill', inplace=True)
#no Excel, fizemos a aproximacao de valores com Size='Varies with device' serem trocados por Size='0' para serem nao serem tendenciosos nos calculos
#como sao 1695 valores com Size='Varies with device', o que representa 0.02% do conjunto de dados, entao e razoavel fazer essa aproximacao pois nao afetara de maneira significativa nos calculos

#transforma o conjunto de Rating para inteiros para a aplicacao do modelo de Árvore de Decisão
df['Rating'] = df['Rating'].astype(int)
#motivacao: como queremos responder "O que faz os usuários gostarem dos aplicativos?" nao precisamos prever com precisao em casa decimais os valores de Rating basta saber se as notas sao boas, medias ou ruins

df['Installs'] = df['Installs'].str.replace(',','') #transforma de str para int, para fins de calculo
df['Installs'] = df['Installs'].str.replace('+','') 

#informacoes uteis sobre a base de dados, avaliando atributos
df['Current Ver'].value_counts()
df['Type'].value_counts()

df['Genres'].nunique()
df['Content Rating'].nunique()

df['Size'].max()
df['Size'].nunique()
df['Size'].mode()
df['Size'].value_counts()

df['Category'].nunique()
df['Category'].value_counts()
y = df.groupby(['Category'])['Rating'].mean()
y.sort_values(ascending=False, inplace=True)

#devido a problemas para a transformacao de variaveis categoricas para numericas de alguns atributos, foi escolhido somente atributos considerados relevantes, atraves de analise qualitativa
#fatia a base em informacoes que foram consideradas relevantes em uma primeira analise qualitativa dos dados
df = df[['Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Content Rating', 'Genres']]

#reordenamento de atributos para fins de calculo
df0 = df.columns.tolist()
df0 = df0[1:] + df0[:1]
df = df[df0]

#define novos conjuntos de dados para serem trabalhados
x = df.iloc[:,[0,1]].values

#escalonamento dos dados
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
kmeans = KMeans(n_clusters = 4, random_state = 0)
previsoes = kmeans.fit_predict(x)

#visualizacao grafica da clusterizacao n=4
plt.figure(figsize=(10,7))
plt.scatter(x[previsoes == 0, 0], x[previsoes == 0, 1], s = 50, c = 'green', label = 'Cluester 1')
plt.scatter(x[previsoes == 1, 0], x[previsoes == 1, 1], s = 50, c = 'blue', label = 'Cluester 2')
plt.scatter(x[previsoes == 2, 0], x[previsoes == 2, 1], s = 50, c = 'red', label = 'Cluester 3')
plt.scatter(x[previsoes == 3, 0], x[previsoes == 3, 1], s = 50, c = 'orange', label = 'Cluester 4')
plt.xlabel('Rating')
plt.ylabel('Reviews')
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
plt.savefig('kmeans5.png')
plt.legend()

#redefinindo o dataframe para incluir app e translated review novamente
df = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Numera_PS/googleplaystore_user_reviews.csv', engine='python')
df.dropna(inplace=True)

#junta a base de dados com os resultados obtidos da clusterizacao
nova_class = np.column_stack((df, previsoes))
nova_class = nova_class[nova_class[:,5].argsort()]
nova_class = pd.DataFrame(data=nova_class)



