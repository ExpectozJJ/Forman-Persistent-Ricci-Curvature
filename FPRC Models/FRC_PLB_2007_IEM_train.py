import numpy as np
import networkx as nx
import scipy as sp
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import math
import multiprocessing as mp
import sys 
import os 
import glob 

path1 = "ORC PLB IEM 2007/forman_train_graphs/"
path2 = "ORC PLB IEM 2007/forman_test_graphs/"

if not os.path.exists("ORC PLB IEM 2007"):
    os.mkdir("ORC PLB IEM 2007")
if not os.path.exists("ORC PLB IEM 2007/PQR/"):
    os.mkdir("ORC PLB IEM 2007/PQR/")
if not os.path.exists("ORC PLB IEM 2007/MOL/"):
    os.mkdir("ORC PLB IEM 2007/MOL/")
if not os.path.exists("ORC PLB IEM 2007/Complexes/"):
    os.mkdir("ORC PLB IEM 2007/Complexes/")
if not os.path.exists(path1):
    os.mkdir(path1)
if not os.path.exists(path2):
    os.mkdir(path2)

# load GraphRicciCuravture package
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

from collections import defaultdict

eleid = {'C':1, 'N':2, 'O':3, 'S':4, 'P':5, 'F':6, 'CL':7, 'BR':8, 'I':9}
vdw = {'C':1.20, 'N':1.55, 'O':1.52, 'S':1.80, 'P':1.80, 'F':1.47, 'CL':1.75, 'BR':1.85, 'I':1.98}
aa_list = ['ALA','ARG','ASN','ASP','CYS','GLU','GLN','GLY','HIS','HSE','HSD','SEC',
           'ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','PYL']

a, b = int(sys.argv[1]), int(sys.argv[2])

list2 = np.load("ORC PLB IEM 2007/train_list.npz", allow_pickle=True)["arr_0"]
test_list = np.load("ORC PLB IEM 2007/test_list.npz", allow_pickle=True)["arr_0"]

def listprep(name1, name2):
    f = open(name1, 'r')
    g = open(name2, 'r')
    contents = f.readlines()
    contents1 = g.readlines()

    refined_set = []
    for i in range(5,len(contents)):
        refined_set = np.append(refined_set, contents[i][:4])
    
    core_set = []
    for i in range(5,len(contents1)):
        core_set = np.append(core_set, contents1[i][:4])

    training_set = list(set(refined_set).symmetric_difference(core_set))
    test_set = core_set

    train_BindingAffinity = []
    test_BindingAffinity = []
    for i in range(len(training_set)):
        for j in range(6, len(contents)):
            if contents[j][:4]==training_set[i]:
                #print(training_set[i], contents[j][19:23])
                train_BindingAffinity = np.append(train_BindingAffinity, float(contents[j][18:23]))
                
    for i in range(len(test_set)):
        for j in range(6, len(contents)):
            if contents[j][:4]==test_set[i]:
                #print(training_set[i], contents[j][19:23])
                test_BindingAffinity = np.append(test_BindingAffinity, float(contents[j][18:23]))

    np.save("ORC PLB IEM 2007/train_BindingAffinity.npy", train_BindingAffinity)
    np.save("ORC PLB IEM 2007/test_BindingAffinity.npy", test_BindingAffinity)
    print("Y_train and Y_test Data have been created. :P")
    return [training_set, test_set, refined_set]

def create_test(threshold, num_of_gaps):
    # Creates a feature vector of x with dimensions: number of PDBIDs x 36 combinations x 100 x 20.
    #print("Creating Test Set...")
    #feature_x = []

    test_list = np.load("ORC PLB IEM 2007/test_list.npz", allow_pickle=True)
    for i in range(a, b+1):
        pdbid = test_list["arr_0"][i]
        print("Processing: ", pdbid)
        data = np.load("ORC PLB IEM 2007/Complexes/"+pdbid+"_complex.npz", allow_pickle=True)
        pro_data, lig_data = data['PRO'], data['LIG']

        pro_hvy_atom = ['H', 'C', 'N', 'O', 'S']
        lig_hvy_atom = ['H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
        pro_coords, lig_coords = [], []
        pro_q, lig_q = [], []
        for p in pro_hvy_atom:
            temp = []
            temp_q = []
            for proatm in pro_data:
                for j in range(len(proatm['typ'])):
                    pt = str(proatm['typ'][j]).replace(" ", ""); pt = pt.upper()
                    if pt == p:
                        temp.append(proatm['pos'][j])
                        temp_q.append(proatm['charge'][j])
            pro_coords.append(temp)
            pro_q.append(temp_q)
            
        for q in lig_hvy_atom:
            temp = []
            temp_q = []
            for ligatm in lig_data:
                for j in range(len(ligatm['typ'])):
                    lt = str(ligatm['typ'][j]).replace(" ", ""); lt = lt.upper()
                    if lt == q:
                        temp.append(ligatm['pos'][j])
                        temp_q.append(ligatm['charge'][j])
            lig_coords.append(temp)
            lig_q.append(temp_q)

        X_graphs = []
        for i in range(0, len(pro_coords)):
            for j in range(0, len(lig_coords)):
                temp_feature = []
                #print("\n")
                # generate graph with protein and ligand atom combination and specified cutoff distance
                for l in np.linspace(0, threshold, num_of_gaps):
                #l = 7.4
                    #print(pro_hvy_atom[i],"--",lig_hvy_atom[j], ": ", l, end="\r")
                    G = nx.Graph()
                    G = gen_graph(pro_coords[i], lig_coords[j], pro_hvy_atom[i], lig_hvy_atom[j], pro_q[i], lig_q[j], l)
                    #vertices = np.zeros(G.number_of_nodes())
                    if G.number_of_edges() > 0:
                        frc = FormanRicci(G)
                        frc.compute_ricci_curvature()
                        temp_feature.append(frc.G)
                    else:
                        temp_feature.append(G)
                    
                    """
                        for v in orc.G.nodes():
                            try:
                                vertices[v] = orc.G.nodes[v]["ricciCurvature"]
                            except:
                                vertices[v] = 0
                                
                    # Binning the vertex ORCs
                    index = np.linspace(-1, 1, num_of_bins)
                    bins = np.zeros(num_of_bins)
                    for v in range(len(vertices)):
                        for ind in range(len(index)-1):
                            if vertices[v] >= index[ind] and vertices[v] < index[ind+1]:
                                bins[ind] += 1
                    temp_feature.append(bins)
                    """
                X_graphs.append(temp_feature)
        #feature_x.append(X_feature)
        np.save("ORC PLB IEM 2007/forman_test_graphs/"+pdbid+"_"+str(threshold)+".npy", X_graphs)

def create_train(threshold, num_of_gaps):
    # Creates a feature vector of x with dimensions: number of PDBIDs x 36 combinations x 100 x 20.
    #print("Creating Training Set...")
    #feature_x = []

    train_list = np.load("ORC PLB IEM 2007/train_list.npz", allow_pickle=True)
    for i in range(a, b+1):
        pdbid = train_list["arr_0"][i]
        print("Processing: ", pdbid)
        data = np.load("ORC PLB IEM 2007/Complexes/"+pdbid+"_complex.npz", allow_pickle=True)
        pro_data, lig_data = data['PRO'], data['LIG']

        pro_hvy_atom = ['H', 'C', 'N', 'O', 'S']
        lig_hvy_atom = ['H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
        pro_coords, lig_coords = [], []
        pro_q, lig_q = [], []
        for p in pro_hvy_atom:
            temp = []
            temp_q = []
            for proatm in pro_data:
                for j in range(len(proatm['typ'])):
                    pt = str(proatm['typ'][j]).replace(" ", ""); pt = pt.upper()
                    if pt == p:
                        temp.append(proatm['pos'][j])
                        temp_q.append(proatm['charge'][j])
            pro_coords.append(temp)
            pro_q.append(temp_q)
            
        for q in lig_hvy_atom:
            temp = []
            temp_q = []
            for ligatm in lig_data:
                for j in range(len(ligatm['typ'])):
                    lt = str(ligatm['typ'][j]).replace(" ", ""); lt = lt.upper()
                    if lt == q:
                        temp.append(ligatm['pos'][j])
                        temp_q.append(ligatm['charge'][j])
            lig_coords.append(temp)
            lig_q.append(temp_q)

        X_graphs = []
        for i in range(0, len(pro_coords)):
            for j in range(0, len(lig_coords)):
                temp_feature = []
                # generate graph with protein and ligand atom combination and specified cutoff distance
                for l in np.linspace(0, threshold, num_of_gaps):
                #l = 7.4
                    #print(pro_hvy_atom[i],"--",lig_hvy_atom[j], ": ", l, end="\r")
                    G = nx.Graph()
                    G = gen_graph(pro_coords[i], lig_coords[j], pro_hvy_atom[i], lig_hvy_atom[j], pro_q[i], lig_q[j], l)
                    #vertices = np.zeros(G.number_of_nodes())
                    if G.number_of_edges() > 0:
                        frc = FormanRicci(G)
                        frc.compute_ricci_curvature()
                        temp_feature.append(frc.G)
                    else:
                        temp_feature.append(G)
                    
                    """
                        for v in orc.G.nodes():
                            try:
                                vertices[v] = orc.G.nodes[v]["ricciCurvature"]
                            except:
                                vertices[v] = 0
                                
                    # Binning the vertex ORCs
                    index = np.linspace(-1, 1, num_of_bins)
                    bins = np.zeros(num_of_bins)
                    for v in range(len(vertices)):
                        for ind in range(len(index)-1):
                            if vertices[v] >= index[ind] and vertices[v] < index[ind+1]:
                                bins[ind] += 1
                    temp_feature.append(bins)
                    """
                X_graphs.append(temp_feature)
        #feature_x.append(X_feature)
        np.save("ORC PLB IEM 2007/forman_train_graphs/"+pdbid+"_"+str(threshold)+".npy", X_graphs)

def gbt_rf(threshold, cutoff):
    X_train = np.load("ORC PLB IEM 2007/train_x_"+str(cutoff)+"_"+str(threshold)+".npy", allow_pickle=True)
    X_test = np.load("ORC PLB IEM 2007/test_x_"+str(cutoff)+"_"+str(threshold)+".npy", allow_pickle=True)
    Y_test = np.load("ORC PLB IEM 2007/test_BindingAffinity.npy", allow_pickle=True)
    Y_train = np.load("ORC PLB IEM 2007/train_BindingAffinity.npy", allow_pickle=True)

    gbt_e, rf_e = [], []

    #print("Y Training Values...")
    #print(Y_train)
    #print("Y Test Values...")
    #print(Y_test)

    print("Predicting via Exponential Kernel...")
    print("Performing GBT...")
    for i in range(10):
    # GBT
        params={'n_estimators': 20000, 'max_depth': 8, 'min_samples_split': 2,
                'learning_rate': 0.001, 'loss': 'ls','max_features':'sqrt','subsample':0.7}
        clf = ensemble.GradientBoostingRegressor(**params)
        clf.fit(X_train, Y_train)
        mse = mean_squared_error(Y_test, clf.predict(X_test))
        print("RMSE: %.4f" % math.sqrt(mse))
        pearcorr = sp.stats.pearsonr(Y_test, clf.predict(X_test))
        gbt_e.append(pearcorr[0])
    print("PCC: ", np.median(gbt_e))

    print("Performing RF...")
    for i in range(10):
    # Random forest
        regr = ensemble.RandomForestRegressor(n_estimators = 1000,max_features='auto')
        regr.fit(X_train,Y_train)
        mse = mean_squared_error(Y_test, regr.predict(X_test))
        print("RMSE: %.4f" % math.sqrt(mse))
        pearcorr = sp.stats.pearsonr(Y_test, regr.predict(X_test))
        rf_e.append(pearcorr[0])
    print("PCC: ", np.median(rf_e))
    return [np.median(gbt_e), np.median(rf_e)]

def create_complex(filename):
    pro_data = np.load("ORC PLB IEM 2007/PQR/"+filename+"_pocket.npz", allow_pickle=True)
    lig_data = np.load("ORC PLB IEM 2007/MOL/"+filename+"_ligand.npz", allow_pickle=True)

    pro_data = pro_data['PRO']
    lig_data = lig_data['LIG']

    pro_hvy_atom = ['H', 'C', 'N', 'O', 'S']
    lig_hvy_atom = ['H', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']

    for pro in pro_data:
        pro_atm = pro['typ']
        pro_coords = pro['pos']
        pro_q = pro['charge']

    for lig in lig_data:
        lig_atm = lig['typ']
        lig_coords = lig['pos']
        lig_q = lig['charge']
    
    data = dict()
    lig_atm1 = []
    lig_coords1 = []
    lig_q1 = []
    for i in range(len(lig_atm)):
        if lig_atm[i].replace(" ", "") in lig_hvy_atom:
            lig_atm1 = np.append(lig_atm1, lig_atm[i])
            lig_q1 = np.append(lig_q1, lig_q[i])
            if len(lig_coords1)==0:
                lig_coords1 = [lig_coords[i]]
            else:
                lig_coords1 = np.append(lig_coords1, [lig_coords[i]], axis = 0)

    #print(lig_coords)

    pro_atm1 = []
    pro_coords1 = []
    pro_q1 = []
    for i in range(len(pro_coords)):
        #print(pro_coords[i], lig_coords1[k], dist)
        if pro_atm[i].replace(" ", "") in pro_hvy_atom: # PDB Pocket Data (No need to cut protein)
            pro_atm1 = np.append(pro_atm1, pro_atm[i])
            pro_q1 = np.append(pro_q1, pro_q[i])
            if len(pro_coords1)==0:
                pro_coords1 = [pro_coords[i]]
            else:
                pro_coords1 = np.append(pro_coords1, [pro_coords[i]], axis = 0)

    data['PRO']=[{'typ': pro_atm1, 'pos': pro_coords1, 'charge': pro_q1}]
    data['LIG']=[{'typ': lig_atm1, 'pos': lig_coords1, 'charge': lig_q1}]
    np.savez("ORC PLB IEM 2007/Complexes/"+filename+"_complex.npz", **data)

def convertpdb(filename):
    f=open(filename, "r")
    if f.mode == 'r':
        contents = f.readlines()
    
    #recordname = []

    #atomNum = []
    #altLoc = []
    #resName = []

    #chainID = []
    #resNum = []
    X = []
    Y = []
    Z = []

    #occupancy = []
    #betaFactor = []
    element = []
    charge = []
    
    
    for i in range(len(contents)):
        thisLine = contents[i]

        if thisLine[0:4]=='ATOM' and thisLine[17:20] in aa_list:
            #recordname = np.append(recordname,thisLine[:6].strip())
            #atomNum = np.append(atomNum, float(thisLine[6:11]))
            #altLoc = np.append(altLoc,thisLine[16])
            #resName = np.append(resName, thisLine[17:20].strip())
            #chainID = np.append(chainID, thisLine[21])
            #resNum = np.append(resNum, float(thisLine[23:26]))
            X = np.append(X, float(thisLine[30:38]))
            Y = np.append(Y, float(thisLine[38:46]))
            Z = np.append(Z, float(thisLine[46:54]))
            #occupancy = np.append(occupancy, float(thisLine[55:60]))
            #betaFactor = np.append(betaFactor, float(thisLine[61:66]))
            element = np.append(element,thisLine[12:14])

    #print(atomName)
    a = {'PRO': [{'typ': element, 'pos': np.transpose([X,Y,Z])}]}
    np.savez(filename[:-4]+".npz", **a)

def convertmol2(filename):
    f=open(filename, "r")
    if f.mode == 'r':
        contents = f.readlines()
    
    linesize = len(contents)
    
    X = [] 
    Y = []
    Z = []
    atomName = []
    charge = []

    for i in range(linesize):
        if contents[i][0:13] == '@<TRIPOS>ATOM':
            ligstart = i+1
        if contents[i][0:13] == '@<TRIPOS>BOND':
            ligend = i-1
    
    for i in range(ligstart, ligend+1):
        line = contents[i]
        if line[8:17] == 'thiophene':
            typ = line[48:50]
            if not typ[1].isalpha():
                typ = ' '+typ[0]
            x = float(line[18:27])
            y = float(line[27:37])
            z = float(line[37:47])
            q = float(line[69:75])
        else:
            typ = line[47:49]
            if not typ[1].isalpha():
                typ = ' '+typ[0]
            x = float(line[17:26])
            y = float(line[26:36])
            z = float(line[36:46])
            q = float(line[70:76])

        X = np.append(X, x)
        Y = np.append(Y, y)
        Z = np.append(Z, z)
        atomName = np.append(atomName, typ)
        charge = np.append(charge, q)
    
    data = {'LIG': [{'typ': atomName, 'pos': np.transpose([X,Y,Z]), 'charge': charge}]}
    np.savez(filename[:-5]+".npz", **data)

def convertpqr(filename):
    f=open(filename, "r")
    if f.mode == 'r':
        contents = f.readlines()
    
    #recordname = []

    #atomNum = []
    #altLoc = []
    #resName = []

    #chainID = []
    #resNum = []
    X = []
    Y = []
    Z = []

    #occupancy = []
    #betaFactor = []
    element = []
    charge = []
    
    
    for i in range(len(contents)):
        thisLine = contents[i]

        if thisLine[0:4]=='ATOM' and thisLine[17:20] in aa_list:
            #recordname = np.append(recordname,thisLine[:6].strip())
            #atomNum = np.append(atomNum, float(thisLine[6:11]))
            #altLoc = np.append(altLoc,thisLine[16])
            #resName = np.append(resName, thisLine[17:20].strip())
            #chainID = np.append(chainID, thisLine[21])
            #resNum = np.append(resNum, float(thisLine[23:26]))
            X = np.append(X, float(thisLine[30:38]))
            Y = np.append(Y, float(thisLine[38:46]))
            Z = np.append(Z, float(thisLine[46:54]))
            charge = np.append(charge, float(thisLine[55:62]))
            #betaFactor = np.append(betaFactor, float(thisLine[61:66]))
            element = np.append(element,thisLine[12:14])

    #print(atomName)
    a = {'PRO': [{'typ': element, 'pos': np.transpose([X,Y,Z]), 'charge': charge}]}
    np.savez(filename[:-4]+".npz", **a)

def convertall():
    print("Converting Raw Data and Creating Complexes...")
    
    for i in range(len(list2)):
        #print(list2[i])
        convertpqr("ORC PLB IEM 2007/PQR/"+list2[i]+"_pocket.pqr")
        convertmol2("ORC PLB IEM 2007/MOL/"+list2[i]+"_ligand.mol2")
        create_complex(list2[i])
    
    for i in range(len(test_list)):
        convertpqr("ORC PLB IEM 2007/PQR/"+test_list[i]+"_pocket.pqr")
        convertmol2("ORC PLB IEM 2007/MOL/"+test_list[i]+"_ligand.mol2")
        create_complex(test_list[i])

    print("All Complexes Created")

def gen_graph(pro_pos, lig_pos, pro_typ, lig_typ, pro_q, lig_q, cutoff):
    G = nx.Graph()

    for i in range(len(pro_pos)):
        G.add_node(i, atom = pro_typ, coords = pro_pos[i])
        
    for j in range(len(lig_pos)):
        ind = G.number_of_nodes()
        G.add_node(ind, atom = lig_typ, coords = lig_pos[j])
        
    for i in range(len(pro_pos)):
        for j in range(len(lig_pos)):
            q1, q2 = pro_q[i], lig_q[j]
            ind = len(pro_pos)+j
            dist = np.linalg.norm(pro_pos[i]-lig_pos[j])
            dist = 1/(1+math.exp(-(100*q1*q2)/dist))
            if dist <= cutoff:
                G.add_edge(i, ind, weight=1.0) #, weight = dist)
    return G

#convertall()
#create_train(10)
#create_test(10)

"""
if __name__ == "__main__":
    
    for i in range(len(list2)):
        #print(list2[i])
        convertpqr("ORC PLB IEM 2007/PQR/"+list2[i]+"_pocket.pqr")
        convertmol2("ORC PLB IEM 2007/MOL/"+list2[i]+"_ligand.mol2")
        create_complex(list2[i])
    
    for i in range(len(test_list)):
        convertpqr("ORC PLB IEM 2007/PQR/"+test_list[i]+"_pocket.pqr")
        convertmol2("ORC PLB IEM 2007/MOL/"+test_list[i]+"_ligand.mol2")
        create_complex(test_list[i])

    print("All Complexes Created")
    create_test(1, 100)
    create_train(1,100)
"""

create_train(1, 100)
