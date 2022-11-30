from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import re
import pickle
import json
import warnings
warnings.filterwarnings('ignore')
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import StandardScaler

# читаем записанные параметры модели
with open('../pickle/medians.pickle', 'rb') as f:          # медианы для заполнения na
    medians = pickle.load(f)
with open('../pickle/scaler.pickle', 'rb') as f:           # StandardScaler
    scaler = pickle.load(f)
with open('../pickle/encoder.pickle', 'rb') as f:          # OneHotEncoder
    encoder = pickle.load(f)
with open('../pickle/model_weights.pickle', 'rb') as f:    # веса модели
    model = pickle.load(f)

# чтение csv-файла с несколькими объектами 
df_test_m = pd.read_csv('data/cars_test.csv')

# чтение json-файла с одним объектом 
with open('data/car_test.json') as f:
    json_test_s = json.load(f)
df_test_s = pd.DataFrame(columns=json_test_s.keys())
df_test_s.loc[len(df_test_s)] = list(json_test_s.values())

def get_prediction(df):
    '''
        Returns the dataframe with the prediction column
    '''

    df_raw = df.copy()
    def get_numerical_mileage(row):
        '''
            Returns mileage in the kmpl units
                On average 1 liter of gas weights 0.73 kg -> 1 kg = 1.37 liter
        '''
        if row == row:
            mileage = float(row.split()[0]) if 'kmpl' in row else float(row.split()[0]) * 1.37
            return mileage
        else:
            return row
        

    df['mileage'] = df['mileage'].apply(get_numerical_mileage)
    df['engine'] = df['engine'].apply(lambda row: float(row.split()[0]) if row == row else row)
    df['max_power'] = df['max_power'].apply(lambda row: float(row.split()[0]) if row == row else row)

    def get_numerical_torque_and_max_torque_rpm(row):
        '''
            Returns torque in the nm units and max_torque_rpm in rpm units
                On average 1 kgm = 9.8 nm
        '''
        if row == row:
            row = re.sub(',', '', row).lower()
            pat = '\d+\.*\d*'
            torque = float(re.findall('\d+\.*\d*', row)[0])
            if 'kgm' in row:
                torque *= 9.8
            max_torque_rpm = np.NaN
            if len(re.findall(pat, row)) == 2:
                max_torque_rpm = float(re.findall(pat, row)[1])
            elif len(re.findall(pat, row)) == 3: 
                    max_torque_rpm = float(re.findall(pat, row)[1])
                    max_torque_rpm = 0.5*float(re.findall(pat, row)[1]) + 0.5*float(re.findall(pat, row)[2])
            return [torque, max_torque_rpm]
        else:
            return [np.NaN, np.NaN]

    df['torque_and_max_torque_rpm'] = df['torque'].apply(get_numerical_torque_and_max_torque_rpm)
    df['torque'] = df['torque_and_max_torque_rpm'].apply(lambda row: row[0])
    df['max_torque_rpm'] = df['torque_and_max_torque_rpm'].apply(lambda row: row[1])
    df = df.drop('torque_and_max_torque_rpm', axis='columns')

    for c in ['mileage', 'engine', 'max_power', 'torque', 'seats', 'max_torque_rpm']:
        df[c] = df[c].fillna(medians[c])

    df['engine'] = df['engine'].astype(int)
    df['seats'] = df['seats'].astype(int)

    df = df.drop(['selling_price'], axis='columns').copy()
    df['name'] = df['name'].apply(lambda s: s.split()[0] + s.split()[1])

    cols_to_encode = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']
    df_sp = encoder.transform(df[cols_to_encode])
    df = pd.concat([df, pd.DataFrame.sparse.from_spmatrix(df_sp)], axis=1)
    df = df.drop(cols_to_encode, axis='columns')

    df = scaler.transform(df)

    pred = model.predict(df)
    df_raw['predicted_price'] = pred

    return df_raw


app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

@app.get('/')
async def root():
    return {'message': 'Car price predict'}

# тест одного объекта
# {
#   "name": "Mahindra Xylo E4 BS IV",
#   "year": 2010,
#   "selling_price": 229999,
#   "km_driven": 168000,
#   "fuel": "Diesel",
#   "seller_type": "Individual",
#   "transmission": "Manual",
#   "owner": "First Owner",
#   "mileage": "14.0 kmpl",
#   "engine": "2498 CC",
#   "max_power": "112 bhp",
#   "torque": "260 Nm at 1800-2200 rpm",
#   "seats": 7
# }
@app.post("/predict_item")
async def predict_item(item: Item) -> float:
    df_test_s = pd.DataFrame(columns=dict(item).keys())
    df_test_s.loc[len(df_test_s)] = list(dict(item).values())
    return get_prediction(df_test_s)['predicted_price'].iloc[0]


# тест нескольких объектов
# [{
#   "name": "Mahindra Xylo E4 BS IV",
#   "year": 2010,
#   "selling_price": 229999,
#   "km_driven": 168000,
#   "fuel": "Diesel",
#   "seller_type": "Individual",
#   "transmission": "Manual",
#   "owner": "First Owner",
#   "mileage": "14.0 kmpl",
#   "engine": "2498 CC",
#   "max_power": "112 bhp",
#   "torque": "260 Nm at 1800-2200 rpm",
#   "seats": 7
# },
# {
#   "name": "Tata Nexon 1.5 Revotorq XE",
#   "year": 2017,
#   "selling_price": 665000,
#   "km_driven": 25000,
#   "fuel": "Diesel",
#   "seller_type": "Individual",
#   "transmission": "Manual",
#   "owner": "First Owner",
#   "mileage": "21.5 kmpl",
#   "engine": "1497 CC",
#   "max_power": "108.5 bhp",
#   "torque": "260Nm@ 1500-2750rpm",
#   "seats": 5
# }]
@app.post("/predict_items")
async def predict_items(items: List[Item]) -> List[float]:
    df_test_m = pd.DataFrame(columns=dict(items[0]).keys())
    for item in items:
        df_test_m.loc[len(df_test_m)] = list(dict(item).values())
    return list(get_prediction(df_test_m)['predicted_price'])





