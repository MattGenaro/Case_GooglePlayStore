import pandas as pd
import numpy as np

df = pd.read_csv('C:/googleplaystore_user_reviews.csv', engine='python')

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

dfp = df[df.Sentiment=='Positive'].Translated_Review
mcp = pd.Series(' '.join(dfp).lower().split()).value_counts()[:100]

dfneg = df[df.Sentiment=='Negative'].Translated_Review
mcnn = pd.Series(' '.join(dfneg).lower().split()).value_counts()[:100]

dfnt = df[df.Sentiment=='Neutral'].Translated_Review
mcn = pd.Series(' '.join(dfnt).lower().split()).value_counts()[:100]

from pandas import DataFrame
x = pd.DataFrame(data=mcp, columns=['quantidade'])
x['palavras'] = x.index
sx = x['quantidade'].sum()

y = pd.DataFrame(data=mcnn, columns=['quantidade'])
y['palavras'] = y.index
sy = y['quantidade'].sum()

z = pd.DataFrame(data=mcn, columns=['quantidade'])
z['palavras'] = z.index
sz = z['quantidade'].sum()

st = sx + sy+ sz

export_csv = x.to_csv (r'C:\Users\genar\OneDrive\Área de Trabalho\Numera_PS\mostcommonpositive.csv', index = None, header=True)
export_csv = y.to_csv (r'C:\Users\genar\OneDrive\Área de Trabalho\Numera_PS\mostcommonnegative.csv', index = None, header=True)
export_csv = z.to_csv (r'C:\Users\genar\OneDrive\Área de Trabalho\Numera_PS\mostcommonneutral.csv', index = None, header=True)



















