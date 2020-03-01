import pandas as pd
import numpy as np
df = pd.read_csv('C:/Users/genar/OneDrive/Área de Trabalho/Numera_PS/googleplaystorecopy.csv', engine='python')
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

df['Installs'] = df['Installs'].str.replace(',','') #substituição de caracteres para transformar a str para int, para fins de calculo
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
x = df.groupby(['Category'])['Rating'].mean()
x.sort_values(ascending=False, inplace=True)

#devido a problemas para a transformacao de variaveis categoricas para numericas de alguns atributos, foi escolhido somente alguns atributos atraves de analise qualitativa
#fatia a base em informacoes que foram consideradas relevantes em uma primeira analise qualitativa dos dados
df = df[['Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Content Rating', 'Genres']]

#reordenamento de atributos para fins de calculo
df0 = df.columns.tolist()
df0 = df0[1:] + df0[:1]
df = df[df0]

#define novos conjuntos de dados para serem trabalhados
previsores = df.iloc[:,1:8].values
classe = df.iloc[:,0].values

#transforma variaveis categoricas como variaveis numericas atraves de rótulos com pesos
from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()
previsores[:,3] = labelencoder_previsores.fit_transform(previsores[:,3])
previsores[:,4] = labelencoder_previsores.fit_transform(previsores[:,4])
previsores[:,5] = labelencoder_previsores.fit_transform(previsores[:,5])

#from sklearn.compose import ColumnTransformer
#transformer = ColumnTransformer(
#   transformers=[("OneHot", OneHotEncoder(),[3,4,5])],remainder='passthrough')
#previsores = transformer.fit_transform(previsores.tolist())
#previsores = previsores.astype('float64')
#One Hot Encoding, para este caso de com o modelo Naive Bayes, se mostrou ineficiente

#escalonamento dos valores
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False)
previsores = scaler.fit_transform(previsores)

#matrizes de correlacao de Pearson para os atributos de Rating, Reviews, Size e Installs
np.corrcoef(classe.astype(float), previsores[:,0].astype(float)) #Rating,  coef 0.08234438
np.corrcoef(classe.astype(float), previsores[:,1].astype(float)) #Rating,  coef 0.09428941
np.corrcoef(classe.astype(float), previsores[:,2].astype(float)) #Rating,  coef 0.09081922
np.corrcoef(classe.astype(float), previsores[:,3].astype(float)) #Rating,  coef 0.06242241
np.corrcoef(classe.astype(float), previsores[:,4].astype(float)) #Rating,  coef -0.00945273
np.corrcoef(classe.astype(float), previsores[:,5].astype(float)) #Rating,  coef 0.00931089
np.corrcoef(previsores[:,0].astype(float), previsores[:,1].astype(float)) #Rating,  coef 0.0654491
np.corrcoef(previsores[:,0].astype(float), previsores[:,3].astype(float)) #Rating,  coef 0.06996263
#a priori, nao foi detectada nenhuma correlacao forte entre Rating e os demais atributos analisados

#método K-Fold para a separação de conujuntos de teste e treino se mostrou uma performance pior e nao foi utilizado
#from sklearn.model_selection import KFold
#from sklearn.svm import SVR
#scores = []
#best_svr = SVR(kernel='rbf')
#kf = KFold(n_splits=10, random_state=42, shuffle=False)
#kf.get_n_splits(previsores)

#for train_index, test_index in kf.split(previsores):
# print('TRAIN:', train_index, 'TEST:', test_index)
# previsores_train, previsores_test = previsores[train_index], previsores[test_index]
# classe_train, classe_test = classe[train_index], classe[test_index]
# best_svr.fit(previsores_train, classe_train)
# scores.append(best_svr.score(previsores_test, classe_test))

#print(np.mean(scores))

#divide os conjuntos de dados entre treino e teste
from sklearn.model_selection import train_test_split
previsores_train, previsores_test, classe_train, classe_test = train_test_split(previsores, classe, test_size=0.3, random_state=0)

#modelo de Naive Bayes
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_train, classe_train)
previsoes = classificador.predict(previsores_test)

#precisao e matriz de confusao
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_test, previsoes)
matriz = confusion_matrix(classe_test, previsoes)

#plot do grafico da matriz de confusao
import seaborn as sn
import matplotlib.pyplot as plt
df_cm = pd.DataFrame(matriz, range(5), range(5))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, xticklabels=['1', '2', '3', '4', '5'], yticklabels=['1', '2', '3', '4', '5'],
           annot=True, annot_kws={"size": 16}, linewidths=.5, fmt="d") 
plt.ylabel("Valores reais das notas médias")
plt.xlabel("Valores previstos das notas médias")
plt.savefig('cmnb.png')
plt.show()

#precisao (label encoder) = 0.17927429274292742
#precisao (label encoder + escalonamento) = 0.1943419434194342
#precisao (label encoder + onehot) = 0.17927429274292742
#precisao (label encoder + onehot + escalonamento) = 0.09225092250922509

#Conclusao modelo Naive Bayes se mostra completamente insuficiente e ineficiente para prever Rating, dados os atributos 
