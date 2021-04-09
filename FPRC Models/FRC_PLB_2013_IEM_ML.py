import glob
import numpy as np
import scipy as sp
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import math
import sys 

#PerFRC Model with Molecular Descriptors 

#num_es = int(sys.argv[1])

fp = int(sys.argv[1])

test_list = np.load("ORC PLB IEM 2013/test_list.npz", allow_pickle=True)["arr_0"]
# Build Test Set for Molecular Descriptors 
test_set = []
cnt = 0
for t in test_list:
    temp = np.load("ORC PLB IEM 2013/forman_test_graphs/"+t+"_1.npy", allow_pickle=True)
    print("Processing: ", t, cnt)
    test_features = []
    for i in range(len(temp)):
        temp_feature = []
        for j in range(0, 10*fp):
            vertices = np.zeros(temp[i][j].number_of_nodes())
            for v in temp[i][j].nodes():
                try:
                    vertices[v] = temp[i][j].nodes[v]["formanCurvature"]
                except:
                    vertices[v] = 0
            
            edges = []
            for (v,w) in temp[i][j].edges():
                try:
                    edges.append(temp[i][j].edges[(v,w)]["formanCurvature"])
                except:
                    edges.append(0)
            #print(edges)
            edges = np.array(edges)

            if len(vertices) > 0:
                mol_feat = []
                mol_feat.append(np.min(vertices))                    # Min
                mol_feat.append(np.max(vertices))                    # Max
                mol_feat.append(np.mean(vertices))                   # Mean
                mol_feat.append(np.std(vertices))                    # Standard Deviation
                #mol_feat.append(mol_feat[1]-mol_feat[0])            # Diameter 
                #mol_feat.append(np.sum(vertices))                   # PerORC graph energy
                pos_sum, pos2, u, v, w = 0, 0, [], 0, 0
                for l in vertices:
                    if l > 0:
                        pos_sum += l
                        pos2 += l*l 
                        u.append(abs(l-mol_feat[2]))
                        v += 1/l
                
                u = np.array(u)
                mol_feat.append(pos_sum)                             # PerORC positive sum 
                #mol_feat.append(neg_sum)                            # PerORC negative sum 
                #mol_feat.append(np.max(np.absolute(vertices)))      # Absolute Max of PerORC
                mol_feat.append(np.sum(u))                           # PerORC Absolute Deviation (PerORC Generalised Graph Energy) 
                #mol_feat.append(np.sum(w))                          # PerORC Mean Absolute Deviation (PerORC Generalised Average Graph Energy) 
                mol_feat.append(np.sum(vertices*vertices))           # PerORC graph energy 2nd moment
                mol_feat.append(pos2)                                # PerORC positive sum 2nd moment
                mol_feat.append(math.log(len(vertices)*v+1))                     # PerORC Pseudo Quasi-Wiener Index
                mol_feat.append(np.sum(u*u*u))                       # PerORC Absolute Deviation 3rd Moment

                if len(edges) > 0:
                    mol_feat.append(np.min(edges))                   # Min
                    mol_feat.append(np.max(edges))                   # Max
                    mol_feat.append(np.mean(edges))                  # Mean
                    mol_feat.append(np.std(edges))                   # Standard Deviation
                    #mol_feat.append(mol_feat[1]-mol_feat[0])        # Diameter 
                    #mol_feat.append(np.sum(edges))                  # PerORC graph energy
                    pos_sum, pos2, u, v, w = 0, 0, [], 0, 0
                    for l in edges:
                        if l > 0:
                            pos_sum += l
                            pos2 += l*l 
                            u.append(abs(l-mol_feat[2]))
                            v += 1/l
                    
                    u = np.array(u)
                    mol_feat.append(pos_sum)                          # PerORC positive sum
                    #mol_feat.append(neg_sum)                         # PerORC negative sum 
                    #mol_feat.append(np.max(np.absolute(edges)))      # Absolute Max of PerORC
                    mol_feat.append(np.sum(u))                        # PerORC Absolute Deviation (PerORC Generalised Graph Energy)
                    #mol_feat.append(np.sum(w))                       # PerORC Mean Absolute Deviation (PerORC Generalised Average Graph Energy)
                    mol_feat.append(np.sum(edges*edges))              # PerORC graph energy 2nd moment
                    mol_feat.append(pos2)                             # PerORC positive sum 2nd moment
                    mol_feat.append(math.log(len(edges)*v+1))                     # PerORC Pseudo Quasi-Wiener Index
                    mol_feat.append(np.sum(u*u*u))                    # PerORC Absolute Deviation 3rd Moment

                else:
                    for z in range(10):
                        mol_feat.append(0)
            else:
                mol_feat = np.zeros(20)
            
            #print(t, mol_feat)
            temp_feature.append(mol_feat)
        test_features.append(temp_feature)
    test_set.append(test_features)
    cnt += 1
np.save("ORC PLB IEM 2013/forman_test_set_"+str(fp)+"_model_4.npy", test_set)

# Build Training Set for Molecular Descriptors 
train_list = np.load("ORC PLB IEM 2013/train_list.npz", allow_pickle=True)["arr_0"]
train_set = []
cnt = 0
for t in train_list:
    temp = np.load("ORC PLB IEM 2013/forman_train_graphs/"+t+"_1.npy", allow_pickle=True)
    print("Processing: ", t, cnt)
    train_features = []
    for i in range(len(temp)):
        temp_feature = []
        for j in range(0, 10*fp):
            vertices = np.zeros(temp[i][j].number_of_nodes())
            for v in temp[i][j].nodes():
                try:
                    vertices[v] = temp[i][j].nodes[v]["formanCurvature"]
                except:
                    vertices[v] = 0
            
            edges = []
            for (v,w) in temp[i][j].edges():
                try:
                    edges.append(temp[i][j].edges[(v,w)]["formanCurvature"])
                except:
                    edges.append(0)

            edges = np.array(edges)
            if len(vertices) > 0:
                mol_feat = []
                mol_feat.append(np.min(vertices))                    # Min
                mol_feat.append(np.max(vertices))                    # Max
                mol_feat.append(np.mean(vertices))                   # Mean
                mol_feat.append(np.std(vertices))                    # Standard Deviation
                #mol_feat.append(mol_feat[1]-mol_feat[0])            # Diameter 
                #mol_feat.append(np.sum(vertices))                   # PerORC graph energy
                pos_sum, pos2, u, v, w = 0, 0, [], 0, 0
                for l in vertices:
                    if l > 0:
                        pos_sum += l
                        pos2 += l*l 
                        u.append(abs(l-mol_feat[2]))
                        v += 1/l
                
                u = np.array(u)
                mol_feat.append(pos_sum)                             # PerORC positive sum
                #mol_feat.append(neg_sum)                            # PerORC negative sum 
                #mol_feat.append(np.max(np.absolute(vertices)))      # Absolute Max of PerORC
                mol_feat.append(np.sum(u))                           # PerORC Absolute Deviation (PerORC Generalised Graph Energy)
                #mol_feat.append(np.sum(w))                          # PerORC Mean Absolute Deviation (PerORC Generalised Average Graph Energy)
                mol_feat.append(np.sum(vertices*vertices))           # PerORC graph energy 2nd moment
                mol_feat.append(pos2)                                # PerORC positive sum 2nd moment
                mol_feat.append(math.log(len(vertices)*v+1))                     # PerORC Pseudo Quasi-Wiener Index
                mol_feat.append(np.sum(u*u*u))                       # PerORC Absolute Deviation 3rd Moment

                if len(edges) > 0:
                    mol_feat.append(np.min(edges))                   # Min
                    mol_feat.append(np.max(edges))                   # Max
                    mol_feat.append(np.mean(edges))                  # Mean
                    mol_feat.append(np.std(edges))                   # Standard Deviation
                    #mol_feat.append(mol_feat[1]-mol_feat[0])        # Diameter 
                    #mol_feat.append(np.sum(edges))                  # PerORC graph energy
                    pos_sum, pos2, u, v, w = 0, 0, [], 0, 0
                    for l in edges:
                        if l > 0:
                            pos_sum += l
                            pos2 += l*l 
                            u.append(abs(l-mol_feat[2]))
                            v += 1/l
                    
                    u = np.array(u)
                    mol_feat.append(pos_sum)                          # PerORC positive sum
                    #mol_feat.append(neg_sum)                         # PerORC negative sum 
                    #mol_feat.append(np.max(np.absolute(edges)))      # Absolute Max of PerORC
                    mol_feat.append(np.sum(u))                        # PerORC Absolute Deviation (PerORC Generalised Graph Energy)
                    #mol_feat.append(np.sum(w))                       # PerORC Mean Absolute Deviation (PerORC Generalised Average Graph Energy)
                    mol_feat.append(np.sum(edges*edges))              # PerORC graph energy 2nd moment
                    mol_feat.append(pos2)                             # PerORC positive sum 2nd moment
                    mol_feat.append(math.log(len(edges)*v+1))                     # PerORC Pseudo Quasi-Wiener Index
                    mol_feat.append(np.sum(u*u*u))                    # PerORC Absolute Deviation 3rd Moment

                else:
                    for z in range(10):
                        mol_feat.append(0)
            else:
                mol_feat = np.zeros(20)

            #print(mol_feat)
            temp_feature.append(mol_feat)
        train_features.append(temp_feature)
    train_set.append(train_features)
    cnt += 1
np.save("ORC PLB IEM 2013/forman_train_set_"+str(fp)+"_model_4.npy", train_set)

X_train = np.load("ORC PLB IEM 2013/forman_train_set_"+str(fp)+"_model_4.npy", allow_pickle=True)
X_test = np.load("ORC PLB IEM 2013/forman_test_set_"+str(fp)+"_model_4.npy", allow_pickle=True)
Y_test = np.load("ORC PLB IEM 2013/test_BindingAffinity.npy", allow_pickle=True)
Y_train = np.load("ORC PLB IEM 2013/train_BindingAffinity.npy", allow_pickle=True)

X_test = np.reshape(X_test, (len(X_test), 50*10*fp*20))
X_train = np.reshape(X_train, (len(X_train), 50*10*fp*20))
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
    print("Filtration Param: ", fp)
    print("GBT RMSE: %.4f" % math.sqrt(mse))
    print("GBT PCC: %.4f" % pearcorr[0])
    gbt_mse.append(pearcorr[0])
    gbt_e.append(math.sqrt(mse))
    gbt_values.append(clf.predict(X_test))
print("Filtration Param: ", fp)
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
    print("Filtration Param: ", fp)
    print("RF RMSE: %.4f" % math.sqrt(mse))
    print("RF PCC: %.4f" % pearcorr[0])
    rf_mse.append(pearcorr[0])
    rf_e.append(math.sqrt(mse))
    rf_values.append(regr.predict(X_test))
print("Filtration Param: ", fp)
print("RF Median PCC: ", np.median(rf_mse))
print("RF Median RMSE: ", np.median(rf_e))
results.append(np.median(rf_mse))
results.append(np.median(rf_e))
rf_values = np.median(rf_values, axis=0)

np.save("ORC PLB IEM 2013/forman_results_"+str(fp)+"_model_4.npy", [results, gbt_values, rf_values])