import glob
import numpy as np
import scipy as sp
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import math
import networkx as nx 
import sys 

#PerORC Model with Molecular Descriptors 

all_comb = [["C", "N", "O", "S", "F", "P", "Cl", "Br", "I"]] #["C", "N"], ["C", "O"], ["C", "N", "O"]]# 
comb_d = []
for c in all_comb:
    comb_d.append(''.join(s for s in c))

test_list = np.load("ORC PLB IDM 2007/test_list.npz", allow_pickle=True)["arr_0"]
train_list = np.load("ORC PLB IDM 2007/train_list.npz", allow_pickle=True)["arr_0"]
# Build Test Set for Molecular Descriptors 

for comb in comb_d:
    test_set = []
    cnt = 0
    for t in test_list:
        [orc_temp, frc_temp] = nx.read_gpickle("ORC PLB IDM 2007/lig_IDM_graphs/"+t+"_15_"+comb+"_ligand.gpickle")
        print("Processing Comb:{}, PDBID:{}, count:{}".format(comb,t, cnt))
        temp_feature = []
        for i in range(len(orc_temp)):
            vertices = np.zeros(orc_temp[i].number_of_nodes())
            for v in orc_temp[i].nodes():
                try:
                    vertices[v] = orc_temp[i].nodes[v]["ricciCurvature"]
                except:
                    vertices[v] = 0
            
            edges = []
            for (v,w) in orc_temp[i].edges():
                try:
                    edges.append(orc_temp[i].edges[(v,w)]["ricciCurvature"])
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
        print(np.shape(temp_feature))
        test_set.append(temp_feature)
        cnt += 1
    np.save("ORC PLB IDM 2007/lig_"+comb+"_2007_ollivier_IDM_test.npy", test_set)

    cnt = 0
    test_set = []
    for t in test_list:
        [orc_temp, frc_temp] = nx.read_gpickle("ORC PLB IDM 2007/lig_IDM_graphs/"+t+"_15_"+comb+"_ligand.gpickle")
        print("Processing Comb:{}, PDBID:{}, count:{}".format(comb,t, cnt))
        temp_feature = []
        for i in range(len(frc_temp)):
            vertices = np.zeros(frc_temp[i].number_of_nodes())
            for v in frc_temp[i].nodes():
                try:
                    vertices[v] = frc_temp[i].nodes[v]["formanCurvature"]
                except:
                    vertices[v] = 0
            
            edges = []
            for (v,w) in frc_temp[i].edges():
                try:
                    edges.append(frc_temp[i].edges[(v,w)]["formanCurvature"])
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
        print(np.shape(temp_feature))
        test_set.append(temp_feature)
        cnt += 1
    np.save("ORC PLB IDM 2007/lig_"+comb+"_2007_forman_IDM_test.npy", test_set)


    train_set = []
    cnt = 0
    for t in train_list:
        [orc_temp, frc_temp] = nx.read_gpickle("ORC PLB IDM 2007/lig_IDM_graphs/"+t+"_15_"+comb+"_ligand.gpickle")
        print("Processing Comb:{}, PDBID:{}, count:{}".format(comb,t, cnt))
        temp_feature = []
        for i in range(len(orc_temp)):
            vertices = np.zeros(orc_temp[i].number_of_nodes())
            for v in orc_temp[i].nodes():
                try:
                    vertices[v] = orc_temp[i].nodes[v]["ricciCurvature"]
                except:
                    vertices[v] = 0
            
            edges = []
            for (v,w) in orc_temp[i].edges():
                try:
                    edges.append(orc_temp[i].edges[(v,w)]["ricciCurvature"])
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
        print(np.shape(temp_feature))
        train_set.append(temp_feature)
        cnt += 1
    np.save("ORC PLB IDM 2007/lig_"+comb+"_2007_ollivier_IDM_train.npy", train_set)

    cnt = 0
    train_set = []
    for t in train_list:
        [orc_temp, frc_temp] = nx.read_gpickle("ORC PLB IDM 2007/lig_IDM_graphs/"+t+"_15_"+comb+"_ligand.gpickle")
        print("Processing Comb:{}, PDBID:{}, count:{}".format(comb,t, cnt))
        temp_feature = []
        for i in range(len(frc_temp)):
            vertices = np.zeros(frc_temp[i].number_of_nodes())
            for v in orc_temp[i].nodes():
                try:
                    vertices[v] = frc_temp[i].nodes[v]["formanCurvature"]
                except:
                    vertices[v] = 0
            
            edges = []
            for (v,w) in frc_temp[i].edges():
                try:
                    edges.append(frc_temp[i].edges[(v,w)]["formanCurvature"])
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
        print(np.shape(temp_feature))
        train_set.append(temp_feature)
        cnt += 1
    np.save("ORC PLB IDM 2007/lig_"+comb+"_2007_forman_IDM_train.npy", train_set)