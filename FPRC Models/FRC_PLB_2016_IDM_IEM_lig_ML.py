import glob
import numpy as np
import scipy as sp
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import math
import sys 

X_train_1 = np.load("ORC PLB IDM 2016/forman_train_set_2016_IDM_model_4.npy", allow_pickle=True)
X_test_1 = np.load("ORC PLB IDM 2016/forman_test_set_2016_IDM_model_4.npy", allow_pickle=True)
X_train_2 = np.load("ORC PLB IEM 2016/forman_train_set_2016_IEM_model_4.npy", allow_pickle=True)
X_test_2 = np.load("ORC PLB IEM 2016/forman_test_set_2016_IEM_model_4.npy", allow_pickle=True)

X_train_3 = np.load("ORC PLB IDM 2016/lig_C_2016_forman_IDM_train.npy", allow_pickle=True)
X_test_3 = np.load("ORC PLB IDM 2016/lig_C_2016_forman_IDM_test.npy", allow_pickle=True)
X_train_4 = np.load("ORC PLB IDM 2016/lig_CN_2016_forman_IDM_train.npy", allow_pickle=True)
X_test_4 = np.load("ORC PLB IDM 2016/lig_CN_2016_forman_IDM_test.npy", allow_pickle=True)
X_train_5 = np.load("ORC PLB IDM 2016/lig_CO_2016_forman_IDM_train.npy", allow_pickle=True)
X_test_5 = np.load("ORC PLB IDM 2016/lig_CO_2016_forman_IDM_test.npy", allow_pickle=True)
X_train_6 = np.load("ORC PLB IDM 2016/lig_CNO_2016_forman_IDM_train.npy", allow_pickle=True)
X_test_6 = np.load("ORC PLB IDM 2016/lig_CNO_2016_forman_IDM_test.npy", allow_pickle=True)
X_train_7 = np.load("ORC PLB IDM 2016/lig_CNOSFPClBrI_2016_forman_IDM_train.npy", allow_pickle=True)
X_test_7 = np.load("ORC PLB IDM 2016/lig_CNOSFPClBrI_2016_forman_IDM_test.npy", allow_pickle=True)

Y_test = np.load("ORC PLB IEM 2016/test_BindingAffinity.npy", allow_pickle=True)
Y_train = np.load("ORC PLB IEM 2016/train_BindingAffinity.npy", allow_pickle=True)

X_test_1 = np.reshape(X_test_1, (len(X_test_1), 36*10*15*20))
X_train_1 = np.reshape(X_train_1, (len(X_train_1), 36*10*15*20))
X_test_2 = np.reshape(X_test_2, (len(X_test_2), 50*10*10*20))
X_train_2 = np.reshape(X_train_2, (len(X_train_2), 50*10*10*20))
X_test_3 = np.reshape(X_test_3, (len(X_test_3), 150*20))
X_train_3 = np.reshape(X_train_3, (len(X_train_3), 150*20))
X_test_4 = np.reshape(X_test_4, (len(X_test_4), 150*20))
X_train_4 = np.reshape(X_train_4, (len(X_train_4), 150*20))
X_test_5 = np.reshape(X_test_5, (len(X_test_5), 150*20))
X_train_5 = np.reshape(X_train_5, (len(X_train_5), 150*20))
X_test_6 = np.reshape(X_test_6, (len(X_test_6), 150*20))
X_train_6 = np.reshape(X_train_6, (len(X_train_6), 150*20))
X_test_7 = np.reshape(X_test_7, (len(X_test_7), 150*20))
X_train_7 = np.reshape(X_train_7, (len(X_train_7), 150*20))

X_test = []
for i in range(len(X_test_1)):
    X_test.append(list(X_test_1[i])+list(X_test_2[i])+list(X_test_3[i])+list(X_test_4[i])+list(X_test_5[i])+list(X_test_6[i])+list(X_test_7[i]))

X_train = []
for i in range(len(X_train_1)):
    X_train.append(list(X_train_1[i])+list(X_train_2[i])+list(X_train_3[i])+list(X_train_4[i])+list(X_train_5[i])+list(X_train_6[i])+list(X_train_7[i]))

print(np.shape(X_train), np.shape(Y_train))
print(np.shape(X_test), np.shape(Y_test))


gbt_mse, rf_mse = [], []
gbt_e, rf_e = [], []
results = []
gbt_values, rf_values = [], []

#print("Y Training Values...")
#print(Y_train)
#print("Y Test Values...")
#print(Y_test)

print("Performing GBT...")
for i in range(10):
# GBT
    params={'n_estimators': 40000, 'max_depth': 7, 'min_samples_split': 2,
            'learning_rate': 0.001, 'loss': 'ls','max_features':'sqrt','subsample':0.7}
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(X_train, Y_train)
    mse = mean_squared_error(Y_test, clf.predict(X_test))
    pearcorr = sp.stats.pearsonr(Y_test, clf.predict(X_test))
    #print("Filtration Param: ", fp)
    print("GBT RMSE: %.4f" % math.sqrt(mse))
    print("GBT PCC: %.4f" % pearcorr[0])
    gbt_mse.append(pearcorr[0])
    gbt_e.append(math.sqrt(mse))
    gbt_values.append(clf.predict(X_test))
#print("Filtration Param: ", fp)
print("GBT Median PCC: ", np.median(gbt_mse))
print("GBT Median RMSE: ", np.median(gbt_e))
results.append(np.median(gbt_mse))
results.append(np.median(gbt_e))
gbt_values = np.median(gbt_values, axis=0)

print("Performing RF...")
for i in range(10):
# Random Forest
    regr = ensemble.RandomForestRegressor(n_estimators = 500,max_features='auto')
    regr.fit(X_train,Y_train)
    mse = mean_squared_error(Y_test, regr.predict(X_test))
    pearcorr = sp.stats.pearsonr(Y_test, regr.predict(X_test))
    #print("Filtration Param: ", fp)
    print("RF RMSE: %.4f" % math.sqrt(mse))
    print("RF PCC: %.4f" % pearcorr[0])
    rf_mse.append(pearcorr[0])
    rf_e.append(math.sqrt(mse))
    rf_values.append(regr.predict(X_test))
#print("Filtration Param: ", fp)
print("RF Median PCC: ", np.median(rf_mse))
print("RF Median RMSE: ", np.median(rf_e))
results.append(np.median(rf_mse))
results.append(np.median(rf_e))
rf_values = np.median(rf_values, axis=0)

np.save("ORC PLB IEM 2016/forman_results_2016_IDM_IEM_lig_multiscale_model_4.npy", [results, gbt_values, rf_values])