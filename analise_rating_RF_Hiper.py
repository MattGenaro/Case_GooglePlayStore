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
x = df.groupby(['Category'])['Rating'].mean()
x.sort_values(ascending=False, inplace=True)

#devido a problemas para a transformacao de variaveis categoricas para numericas de alguns atributos, foi escolhido somente atributos considerados relevantes, atraves de analise qualitativa
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

#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler(feature_range=(0, 1))
#previsores = scaler.fit_transform(previsores)

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

#modelo de floresta aleatoria
from sklearn.model_selection import RandomizedSearchCV
#numeros de arvore na floresta aleatorias
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
#numero de caracteristicas para considerar em cada divisao
max_features = ['auto', 'sqrt']
#numero maximo de niveis em uma arvore
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
#numero minimo de amostras requeridos para dividr um nodo
min_samples_split = [2, 5, 10]
##numero minimo de amostras requeridos para cada folha no nodo
min_samples_leaf = [1, 2, 4]
#metodo de selecao de amostroas para treinar cada arvore
bootstrap = [True, False] # cria uma grade aleatoria
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)

#usa a grade aleatoria para procurar pelos melhores hiperparametros
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
#procura aleatoria por parametros usando validacao cruzada com 3 folds
#procura entre 100 diferentes combinacoes e usa todos os cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(previsores_train, classe_train)

rf_random.best_params_

#parametros estatisticos sobre performance da hiperparametrizacao
def evaluate(model, previsores_test, classe_test):
    predictions = model.predict(previsores_test)
    errors = abs(predictions - classe_test)
    mape = 100 * np.mean(errors / classe_test)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy
    
#modelo base 
base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(previsores_train, classe_train)
base_accuracy = evaluate(base_model, previsores_test, classe_test)

#modelo com melhores valores aleatorios
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, previsores_test, classe_test)

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))


#outro modelo para hiperparametrizacao 
from sklearn.model_selection import GridSearchCV
#cria o parametro de grade baseado nos resultados da busca aleatoria
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

rf = RandomForestRegressor()
#instancia o modelo de busca em grade
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(previsores_train, classe_train)

#melhores parametros para busca em grade
grid_search.best_params_

#parametros estatisticos de performance da hiperparametrizacao
best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, previsores_test, classe_test)

print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))

#precisao do modelo base hiperparametrizado: 75.3131790692907
#precisao do modelo aleatorio hiperparametrizado: 74.72178523372841
#precisao do modelo de busca em grade hiperparametrizado: 75.51581647194953
