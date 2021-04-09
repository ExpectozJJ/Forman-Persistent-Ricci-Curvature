import numpy as np
import networkx as nx
import math
import pickle

# load GraphRicciCuravture package
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

from collections import defaultdict

import os 

#path1 = "ORC PLB IDM 2007/"
path2 = "ORC PLB IDM 2013/"
#path3 = "ORC PLB IDM 2016/"

#if not os.path.exists(path1+"lig_IDM_graphs/"):
    #os.mkdir(path1+"lig_IDM_graphs/")

if not os.path.exists(path2+"lig_IDM_graphs/"):
    os.mkdir(path2+"lig_IDM_graphs/")
#if not os.path.exists(path3+"lig_IDM_graphs/"):
    #os.mkdir(path3+"lig_IDM_graphs/")

def gen_lignet(elem, pdbid, cutoff, path):
    data = np.load(path+"MOL/"+str(pdbid)+"_ligand.npz", allow_pickle=True)
    for d in data["LIG"]:
        lig_pos = d["pos"]
        lig_atm = d["typ"]
    #print(lig_pos, lig_atm)
    G = nx.Graph()
    for j in range(len(lig_atm)):
        if lig_atm[j].strip(" ") in elem:
            #print(lig_atm[j].strip(" "))
            ind = G.number_of_nodes()
            G.add_node(ind, atom = lig_atm[j], coords = lig_pos[j])
            
    for v in G.nodes():
        for w in G.nodes():
            if v!=w:
                dist = np.linalg.norm(G.nodes[v]['coords']-G.nodes[w]['coords'])
                if round(dist,2) <= cutoff:
                    G.add_edge(v, w, weight=1.0)
    return G

all_comb = [["C", "N"], ["C", "O"], ["C", "N", "O"], ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I"]]
comb_d = []
for c in all_comb:
    comb_d.append(''.join(s for s in c))

#train_list_2007 = np.load("ORC PLB IDM 2007/train_list.npz", allow_pickle=True)
#test_list_2007 = np.load("ORC PLB IDM 2007/test_list.npz", allow_pickle=True)
train_list_2013 = np.load("ORC PLB IDM 2013/train_list.npz", allow_pickle=True)
test_list_2013 = np.load("ORC PLB IDM 2013/test_list.npz", allow_pickle=True)
#train_list_2016 = np.load("ORC PLB IDM 2016/train_list.npz", allow_pickle=True)
#test_list_2016 = np.load("ORC PLB IDM 2016/test_list.npz", allow_pickle=True)
"""
for pdbid in train_list_2007["arr_0"]:
    print("Processing v2007: ", pdbid)
    orc_temp = []
    frc_temp = []
    for c in np.linspace(0, 15, 150):
        G = gen_lignet(["C"], pdbid, c, path1)
        if G.number_of_edges() > 0:
            orc = OllivierRicci(G, alpha=0.5)
            orc.compute_ricci_curvature()
            frc = FormanRicci(G)
            frc.compute_ricci_curvature()
            orc_temp.append(orc.G)
            frc_temp.append(frc.G)
        else:
            orc_temp.append(G)
            frc_temp.append(G)

    np.save(path1+"lig_IDM_graphs/"+pdbid+"_ollivier_15_C_ligand.npy", orc_temp)
    np.save(path1+"lig_IDM_graphs/"+pdbid+"_forman_15_C_ligand.npy", frc_temp)
"""

for i in range(len(all_comb)):
    print("Processing: {:20}".format(comb_d[i]))
    for pdbid in train_list_2013["arr_0"]:
        print("Processing v2013: {:4}".format(pdbid))
        orc_temp = []
        frc_temp = []
        for c in np.linspace(0, 15, 150):
            G = gen_lignet(all_comb[i], pdbid, c, path2)
            if G.number_of_edges() > 0:
                orc = OllivierRicci(G, alpha=0.5)
                orc.compute_ricci_curvature()
                frc = FormanRicci(G)
                frc.compute_ricci_curvature()
                orc_temp.append(orc.G)
                frc_temp.append(frc.G)
            else:
                orc_temp.append(G)
                frc_temp.append(G)

        nx.write_gpickle([orc_temp, frc_temp], path2+"lig_IDM_graphs/"+pdbid+"_15_"+comb_d[i]+"_ligand.gpickle")

    for pdbid in test_list_2013["arr_0"]:
        print("Processing v2013: ", pdbid)
        orc_temp = []
        frc_temp = []
        for c in np.linspace(0, 15, 150):
            G = gen_lignet(all_comb[i], pdbid, c, path2)
            if G.number_of_edges() > 0:
                orc = OllivierRicci(G, alpha=0.5)
                orc.compute_ricci_curvature()
                frc = FormanRicci(G)
                frc.compute_ricci_curvature()
                orc_temp.append(orc.G)
                frc_temp.append(frc.G)
            else:
                orc_temp.append(G)
                frc_temp.append(G)

        nx.write_gpickle([orc_temp, frc_temp], path2+"lig_IDM_graphs/"+pdbid+"_15_"+comb_d[i]+"_ligand.gpickle")

"""

for pdbid in train_list_2016["arr_0"]:
    print("Processing v2016: ", pdbid)
    orc_temp = []
    frc_temp = []
    for c in np.linspace(0, 15, 150):
        G = gen_lignet(["C"], pdbid, c, path3)
        if G.number_of_edges() > 0:
            orc = OllivierRicci(G, alpha=0.5)
            orc.compute_ricci_curvature()
            frc = FormanRicci(G)
            frc.compute_ricci_curvature()
            orc_temp.append(orc.G)
            frc_temp.append(frc.G)
        else:
            orc_temp.append(G)
            frc_temp.append(G)

    np.save(path3+"lig_IDM_graphs/"+pdbid+"_ollivier_15_C_ligand.npy", orc_temp)
    np.save(path3+"lig_IDM_graphs/"+pdbid+"_forman_15_C_ligand.npy", frc_temp)
    
"""

