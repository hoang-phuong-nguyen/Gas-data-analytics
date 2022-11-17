import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def reduce_dimension(data, n_comp=4, display=False):
    pca = PCA(n_components=n_comp)
    pca.fit(data.iloc[:, 2:])
    
    if display:
        print("PCA number of features during fitting: ", pca.n_features_in_)
        print("PCA estimated number of components: ", pca.n_components_)
        print("PCA components: \n", pca.components_)
        print("PCA variance: ", pca.explained_variance_)
        print("PCA singular values: ", pca.singular_values_)
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        plt.plot(pca.explained_variance_, 'bs-')
        plt.ylabel("Variance")
        plt.xlabel("Principal component")
        xticks = []
        for i in range(n_comp):
            xticks.append('PC' + str(i+1))
        plt.xticks(np.arange(0, n_comp), xticks)
        plt.grid()
        plt.show()
        
    return pca

def fit_PCA(x_train, x_test, n_comp):
    pca = PCA(n_components=n_comp)
    x_train_new = pca.fit_transform(x_train)
    x_test_new = pca.transform(x_test)
    return x_train_new, x_test_new

def fit_LDA(x_train, y_train, x_test, n_comp):
    lda = LDA(n_components=n_comp)
    x_train_new = lda.fit_transform(x_train, y_train)
    x_test_new = lda.transform(x_test)
    return x_train_new, x_test_new