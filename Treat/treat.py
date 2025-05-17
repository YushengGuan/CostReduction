import pandas as pd
import numpy as np

df = pd.read_excel('ELECSTAT-C_20250517-031311.xlsx')

for i in df.index:
    if str(df.loc[i, 'Country']) != 'nan':
        print(df.loc[i, 'Country'])
        country = df.loc[i, 'Country']
    else:
        df.loc[i, 'Country'] = country
    
    if str(df.loc[i, 'Type']) != 'nan':
        type = df.loc[i, 'Type']
        variable = df.loc[i, 'Variable']
    else:
        df.loc[i, 'Type'] = type
        df.loc[i, 'Variable'] = variable
df.to_excel('Treated.xlsx')
        