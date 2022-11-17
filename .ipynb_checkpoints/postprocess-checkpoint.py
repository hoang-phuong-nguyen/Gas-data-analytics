import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
import seaborn as sns
import plotly.express as px
import json


def plot_raw_data(data, start, end):
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(111)
    data[start:end].plot(y=data.columns[1], color='red', ax=ax)
    data[start:end].plot(y=data.columns[2], color='blue', ax=ax)
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
    # fig.savefig('methane_vs_ethylene.png')

    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(111)
    data[start:end].plot(y='TGS2602-1', ax=ax)
    data[start:end].plot(y='TGS2602-2', ax=ax)
    data[start:end].plot(y='TGS2602-3', ax=ax)
    data[start:end].plot(y='TGS2602-4', ax=ax)
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
    # fig.savefig('TGS2602.png')

    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(111)
    data[start:end].plot(y='TGS2600-1', ax=ax)
    data[start:end].plot(y='TGS2600-2', ax=ax)
    data[start:end].plot(y='TGS2600-3', ax=ax)
    data[start:end].plot(y='TGS2600-4', ax=ax)
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
    # fig.savefig('TGS2600.png')

    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(111)
    data[start:end].plot(y='TGS2610-1', ax=ax)
    data[start:end].plot(y='TGS2610-2', ax=ax)
    data[start:end].plot(y='TGS2610-3', ax=ax)
    data[start:end].plot(y='TGS2610-4', ax=ax)
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
    # fig.savefig('TGS2610.png')

    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(111)
    data[start:end].plot(y='TGS2620-1', ax=ax)
    data[start:end].plot(y='TGS2620-2', ax=ax)
    data[start:end].plot(y='TGS2620-3', ax=ax)
    data[start:end].plot(y='TGS2620-4', ax=ax)
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
    # fig.savefig('TGS2620.png')
    
def postprocess(x_test, y_test, y_predict, gas1_name, gas2_name, display=False):
    target_names = ['Neutral', gas1_name, gas2_name, 'Mixture']
    report = classification_report(y_test, y_predict, target_names=target_names, output_dict=True)    
    conf_mat = confusion_matrix(y_test, y_predict)
    
    # generate output json 
    accuracy_report_json = pd.DataFrame(report).transpose().to_json(orient="split")
    conf_mat_json = pd.DataFrame(conf_mat).to_json(orient="split")

    features = pd.DataFrame(x_test, columns=['Feature 1', 'Feature 2', 'Feature 3'])
    features['Label'] = y_predict
    feature_json = features.to_json(orient="split")
    
    if display:
        print(report)
        sns.heatmap(conf_mat, 
                    annot=True, 
                    annot_kws={"size": 10},
                    cmap='crest', 
                    fmt="d", 
                    xticklabels=target_names,
                    yticklabels=target_names)
        plt.show()
    
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        plot = ax.scatter(x_test[:,0], x_test[:,1], x_test[:,2], c=y_predict)
        plt.legend(handles=plot.legend_elements()[0], labels=target_names)
        plt.show()
    
    return accuracy_report_json, conf_mat_json, feature_json
    # acc_dict = json.loads(accuracy_report_json)
    # conf_dict = json.loads(conf_mat_json)
    # feature_dict = json.loads(feature_json)
    # return acc_dict.update(conf_dict).update(feature_dict)