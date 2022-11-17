import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess(data, fs='1s', display=False):
    # create timedate index 
    new_data = data
    new_data["Time"] = pd.to_timedelta(new_data["Time"], unit='s')
    new_data = new_data.set_index('Time')

    # min-max normalization
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(new_data.values)
    scaled_data = pd.DataFrame(scaled_data, columns=new_data.columns, index=new_data.index)

    # downsampling to 1 second interval
    scaled_data = scaled_data.resample(fs).mean()

    if display: 
        print("Number of samples after downsampling: ", scaled_data.shape[0])
        print(scaled_data.head(10))
    
    return scaled_data

def generate_labels(data, gas1_col, gas2_col, display=False):
    conditions = [
        (data[gas1_col] == 0) & (data[gas2_col] == 0) ,
        (data[gas1_col] > 0) & (data[gas2_col] == 0),
        (data[gas1_col] == 0) & (data[gas2_col] > 0),
        (data[gas1_col] > 0) & (data[gas2_col] > 0),
    ]
    labels = [0, 1, 2, 3]
    data['Label'] = np.select(conditions, labels, default=0)
    
    if display:
        print(data['Label'].value_counts())
    
    return data

def generate_train_test_sets(data, gas1_col, gas2_col, test_set_ratio=0.3, display=False):
    Y = data['Label']
    X = data
    X.drop([gas1_col, gas2_col, 'Label'], axis=1, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_set_ratio, random_state=42)
    
    if display:
        print("Number of samples for training: ", y_train.shape[0])
        print(y_train.value_counts())

        print("Number of samples for testing: ", y_test.shape[0])
        print(y_test.value_counts())
        
    return x_train, x_test, y_train, y_test

def get_gas_names(type):
    if (type == 1):
        fname = "ethylene_methane.txt"
        gas1_name = 'Methane'
        gas2_name = 'Ethylene'
    else: 
        fname = "ethylene_CO.txt"
        gas1_name = "CO"
        gas2_name = "Ethylene"

    gas1_col = gas1_name + '_conc'
    gas2_col = gas2_name + '_conc'
    col_names = ['Time', gas1_col, gas2_col, 
                 'TGS2602-1', 'TGS2602-2', 'TGS2600-1', 'TGS2600-2', 
                 'TGS2610-1', 'TGS2610-2', 'TGS2620-1', 'TGS2620-2',
                 'TGS2602-3', 'TGS2602-4', 'TGS2600-3', 'TGS2600-4', 
                 'TGS2610-3', 'TGS2610-4', 'TGS2620-3', 'TGS2620-4']
    
    return fname, gas1_name, gas2_name, gas1_col, gas2_col, col_names
