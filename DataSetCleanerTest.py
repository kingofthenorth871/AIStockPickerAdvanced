import pandas as pd
import numpy as np

data = pd.read_csv("SelfMadeStockDataset.csv")

print('printer column lengden')
print(len(data.columns))
print(len(data))

mod_data = data.dropna(thresh=40)
mod_data['priceVar1yr'] = mod_data['priceVar1yr'].map(lambda x: x.lstrip('[').rstrip(']'))
mod_data['priceVar1yr'] = mod_data['priceVar1yr'].astype(float)

conditions = [
    mod_data['priceVar1yr'] > 50 , mod_data['priceVar1yr'] < 50
]

choices = [1,0]
mod_data['class'] = np.select(conditions, choices, default=0)
mod_data.drop('date', axis=1, inplace=True)
mod_data.drop('priceVar1yr', axis=1, inplace=True)
print(mod_data)