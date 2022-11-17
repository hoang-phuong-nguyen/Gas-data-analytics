from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as RF

def predict_RF(x_train, y_train, x_test, n_estimators):
    model = RF(n_estimators=n_estimators)
    model = model.fit(x_train, y_train)
    y_predict = model.predict(x_test)    
    return y_predict

def predict_SVM(x_train, y_train, x_test):
    model = svm.SVC(decision_function_shape='ovo')
    model = model.fit(x_train, y_train)
    y_predict = model.predict(x_test)    
    return y_predict