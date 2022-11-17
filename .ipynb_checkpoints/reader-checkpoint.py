import pandas as pd

def load_data(data_path, col_names, display=False):
    data_df = pd.read_csv(data_path, skiprows=1, header=None, delim_whitespace=True)
    data_df.columns = col_names
    
    if display:
        print("Number of data samples: ", data_df.shape[0])
        print(data_df.head(10))
        
    return data_df 