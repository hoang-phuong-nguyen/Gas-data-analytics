import os
import numpy as np
import pandas as pd
from reader import load_data
from postprocess import postprocess, plot_raw_data
from preprocess import preprocess, generate_labels, generate_train_test_sets, get_gas_names
from feature_selection import reduce_dimension, fit_PCA, fit_LDA
from models import predict_RF, predict_SVM 

def load_raw_data(data_path, data_type):
    fname, gas1_name, gas2_name, gas1_col, gas2_col, col_names = get_gas_names(data_type)
    raw_data = load_data(os.path.join(data_path, fname), col_names, display=False)
    # plot_raw_data(raw_data, start=200000, end=500000)

    raw_data_json = raw_data.to_json(orient="split")
    return raw_data_json


def analyze_data(data_json, data_type):
    data = pd.read_json(data_json, orient="split")
    _, gas1_name, gas2_name, gas1_col, gas2_col, _ = get_gas_names(data_type)

    # preprocessing
    processed_data = preprocess(data, fs='1s', display=False)
    # plot_raw_data(processed_data, start=2000, end=5000)

    # generate train, test sets 
    labeled_data = generate_labels(processed_data, gas1_col, gas2_col, display=False)
    x_train, x_test, y_train, y_test = generate_train_test_sets(labeled_data, gas1_col, gas2_col, test_set_ratio=0.3, display=False)

    # feature analysis
    x_train_transformed, x_test_transformed = fit_PCA(x_train, x_test, n_comp=3)
    # x_train_transformed, x_test_transformed = fit_LDA(x_train, y_train, x_test, n_comp=3)

    # classification 
    y_predict = predict_RF(x_train_transformed, y_train, x_test_transformed, n_estimators=15)
    # y_predict = predict_SVM(x_train_transformed, y_train, x_test_transformed)

    accuracy_report_json, conf_mat_json, feature_json = postprocess(x_test_transformed, y_test, y_predict, gas1_name, gas2_name, display=False)
            
    return accuracy_report_json, conf_mat_json, feature_json

if __name__ == '__main__':
    ################################
    # DEFINE INPUT ARGUMENTS HERE! #
    ################################
    data_path = "./data/gas_mixture_dataset"
    data_type = 2 # 1: ethylene + methane; 2: ethylene + CO  

    
    ############################
    # API 1: load_raw_data_api #
    ############################
    # Input arguments: 
    #   1. data_path: MinIO data folder
    #   2. data_type: 1: ethylene + methane; 2: ethylene + CO
    # Output:
    #   1. Number of raw data samples (number) <- from JSON_1: raw data <- FE 
    #   2. Table of raw data samples <- from JSON_1: raw data <- FE 
    #   3. Plot raw data samples: <- from JSON_1: raw data
    #      - IMG 1: concentration rate of 2 gases
    #      - IMG 2: measurements of 4 sensors TGS2600-1, TGS2600-2, TGS2600-3, TGS2600-4 
    #      - IMG 3: measurements of 4 sensors TGS2602-1, TGS2602-2, TGS2602-3, TGS2602-4 
    #      - IMG 4: measurements of 4 sensors TGS2610-1, TGS2610-2, TGS2610-3, TGS2610-4 
    #      - IMG 5: measurements of 4 sensors TGS2620-1, TGS2620-2, TGS2620-3, TGS2620-4 
    raw_data_json = load_raw_data(data_path, data_type)
    
    
    ###########################
    # API 2: analyze_data_api #
    ###########################
    # Input arguments: 
    #   1. raw_data: from API 1 
    #   2. data_type: 1: ethylene + methane; 2: ethylene + CO
    # Output:
    #   1. Number of samples for training and testing <- from JSON_2: classfication results <- FE 
    #   2. Table of classification results <- from JSON_2: classfication results <- FE
    #   3. IMG 6: Confusion matrix <- from JSON_2: classfication results <- FE
    #   4. IMG 7: Classified feature distribution <- from JSON_3: 4 columns (3 axis + data class) <- FE
    accuracy_report_json, conf_mat_json, feature_json = analyze_data(raw_data_json, data_type)
    
    
    #####################################
    # API 3: analyze_unlabeled_data_api #
    #####################################
    # Input arguments: 
    #   1. raw_data: from API 1
    #   2. data_type: 1: ethylene + methane; 2: ethylene + CO
    # Output:
    #   1. Number of samples for training and testing <- from JSON_3: clustering results <- FE 
    #   2. Table of clustering results <- from JSON_4: clustering results <- FE
    #   3. IMG 8: Confusion matrix <- from JSON_4: clustering results <- FE
    #   4. IMG 9: Clustered feature distribution <- from JSON_5: 4 columns (3 axis + data class) <- FE
    # cluster_json = analyze_unlabeled_data(raw_data_json, data_type)
    
