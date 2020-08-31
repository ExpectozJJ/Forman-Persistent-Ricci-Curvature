import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

from frc import GeneralisedFormanRicci

from collections import defaultdict

import collections

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def gen_graph(filename, cutoff):
    data = np.load(filename, allow_pickle = True)
    for item in data['PRO']:
        coords = item['pos']
        atoms = item['atom']
    G = nx.Graph()
    for i in range(len(atoms)):
        G.add_node(i, atom = atoms[i], coords = coords[i])
    for i in range(len(coords)):
        for j in range(len(coords)):
            if i!=j:
                dist = np.linalg.norm(coords[i]-coords[j])
                if round(dist,2) <= cutoff:
                    G.add_edge(i, j) #, weight = dist)
    return G


vals = ["1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M"]
for v in vals:
    raw_output = []
    for id in range(101):
        temp = defaultdict(list)
        s = str(id)
        while len(s) < 4:
            s = '0'+ s
        print("Processing Frame ", id, ": ", v)
        #convertpdb("tmao/md_"+j+"_tmao_4p_tf_"+s+"_OW.pdb")
        data = np.load("tmao/md_"+v+"_tmao_4p_tf_"+s+"_OW.npz", allow_pickle=True)["PRO"]
        #G = gen_graph("tmao/md_"+j+"_tmao_4p_tf_"+s+"_OW.npz", 4)

        for j in data:
            pos = j["pos"]
            labels = {'typ': j["typ"]}
        
        frc = GeneralisedFormanRicci(points=pos, method="rips", labels=labels, epsilon = 4, p = 2)
        print("Simplicial Complex Construction Done.")
        frc_val = frc.compute_forman()
        print("Forman Curvature Computation Done.")

        #frc = FormanRicci(G)
        #frc.compute_ricci_curvature()

        #vertices = np.zeros(3000)
        #for v in frc.G.nodes():
            #try:
                #vertices[v] = frc.G.nodes[v]["formanCurvature"]
            #except:
                #vertices[v] = 0
        
        #raw_output.append(vertices)
        #raw_output.append(list(nx.get_edge_attributes(frc.G, 'formanCurvature').values()))
        temp[0].append(list(frc_val[0].values()))
        temp[1].append(list(frc_val[1].values()))
        temp[2].append(list(frc_val[2].values()))
        raw_output.append(temp)

    np.savez("forman_tmao_raw_op_"+v+".npz", *raw_output)

for v in vals:
    raw_output = []
    for id in range(101):
        temp = defaultdict(list)
        s = str(id)
        while len(s) < 4:
            s = '0'+ s
        print("Processing Frame ", id, ": ", v)
        #convertpdb("tmao/md_"+j+"_tmao_4p_tf_"+s+"_OW.pdb")
        data = np.load("urea/md_"+v+"_urea_4p_tf_"+s+"_OW.npz", allow_pickle=True)["PRO"]
        #G = gen_graph("tmao/md_"+j+"_tmao_4p_tf_"+s+"_OW.npz", 4)

        for j in data:
            pos = j["pos"]
            labels = {'typ': j["typ"]}
        
        frc = GeneralisedFormanRicci(points=pos, method="rips", labels=labels, epsilon = 4, p = 2)
        print("Simplicial Complex Construction Done.")
        frc_val = frc.compute_forman()
        print("Forman Curvature Computation Done.")

        #frc = FormanRicci(G)
        #frc.compute_ricci_curvature()

        #vertices = np.zeros(3000)
        #for v in frc.G.nodes():
            #try:
                #vertices[v] = frc.G.nodes[v]["formanCurvature"]
            #except:
                #vertices[v] = 0
        
        #raw_output.append(vertices)
        #raw_output.append(list(nx.get_edge_attributes(frc.G, 'formanCurvature').values()))
        temp[0].append(list(frc_val[0].values()))
        temp[1].append(list(frc_val[1].values()))
        temp[2].append(list(frc_val[2].values()))
        raw_output.append(temp)

    np.savez("forman_urea_raw_op_"+v+".npz", *raw_output)

def plot_tmao_avg():
    tmao_op = defaultdict(list)
    for v in vals:
        curv_0 = []
        curv_1 = []
        curv_2 = []
        data = np.load("forman_tmao_raw_op_"+v+".npz", allow_pickle=True)
        for d in data.keys():
            curv_0.append(data[d][()][0])
            curv_1.append(data[d][()][1])
            curv_2.append(data[d][()][2])
        tmao_op[v] = [curv_0, curv_1, curv_2]

    avg_op = defaultdict(list)
    for v in vals:
        temp_1 = flatten(tmao_op[v][0])
        temp_2 = flatten(tmao_op[v][1])
        temp_3 = flatten(tmao_op[v][2])
        avg_op[v] = [temp_1, temp_2, temp_3]

    conc = ["1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M"]
    plt.figure(figsize=(10,10))
    plt.rcdefaults()
    for i in range(len(vals)):
        color = plt.cm.jet(i/(len(vals)))
        sns.distplot(avg_op[vals[i]][0], color=color, label = conc[i], kde_kws={'gridsize': 200, 'bw': 0.15} , kde=True, hist=False, norm_hist=True)
    plt.xticks(fontsize=20)
    plt.yticks( fontsize=20)
    plt.legend()
    #plt.axis([-1, 1, 0, 6])
    plt.xlabel("Vertex Curvature", fontsize=20)
    plt.ylabel("Average Density", fontsize=20)
    plt.title("$H_{2}O$ (Tmao) - TIP4P", fontsize=20)
    plt.savefig("forman_avg_tmao_vertex.pdf", dpi=200)
    plt.show()

    conc = ["1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M"]
    plt.figure(figsize=(10,10))
    plt.rcdefaults()
    for i in range(len(vals)):
        color = plt.cm.jet(i/(len(vals)))
        sns.distplot(avg_op[vals[i]][1], color=color, label = conc[i], kde_kws={'gridsize': 200, 'bw': .15} , kde=True, hist=False, norm_hist=True)
    plt.xticks(fontsize=20)
    plt.yticks( fontsize=20)
    plt.legend()
    #plt.axis([-1, 1, 0, 6])
    plt.xlabel("Edge Curvature", fontsize=20)
    plt.ylabel("Average Density", fontsize=20)
    plt.title("$H_{2}O$ (Tmao) - TIP4P", fontsize=20)
    plt.savefig("forman_avg_tmao_edge.pdf", dpi=200)
    plt.show()

    conc = ["1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M"]
    plt.figure(figsize=(10,10))
    plt.rcdefaults()
    for i in range(len(conc)):
        color = plt.cm.jet(i/(len(conc)))
        sns.distplot(avg_op[conc[i]][2], color=color, label = conc[i], kde_kws={'gridsize': 200, 'bw': .15} , kde=True, hist=False, norm_hist=True)
    plt.xticks(fontsize=20)
    plt.yticks( fontsize=20)
    plt.legend()
    #plt.axis([-1, 1, 0, 6])
    plt.xlabel("2-Simplex Curvature", fontsize=20)
    plt.ylabel("Average Density", fontsize=20)
    plt.title("$H_{2}O$ (Tmao) - TIP4P", fontsize=20)
    plt.savefig("forman_avg_tmao_tri.pdf", dpi=200)
    plt.show()

def plot_urea_avg():
    vals = ["1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M"]
    urea_op = defaultdict(list)
    for v in vals:
        curv_0 = []
        curv_1 = []
        curv_2 = []
        data = np.load("forman_urea_raw_op_"+v+".npz", allow_pickle=True)
        for d in data.keys():
            curv_0.append(data[d][()][0])
            curv_1.append(data[d][()][1])
            curv_2.append(data[d][()][2])
        urea_op[v] = [curv_0, curv_1, curv_2]

    avg_op = defaultdict(list)
    for v in vals:
        temp_1 = flatten(urea_op[v][0])
        temp_2 = flatten(urea_op[v][1])
        temp_3 = flatten(urea_op[v][2])
        avg_op[v] = [temp_1, temp_2, temp_3]

    conc = ["1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M"]
    plt.figure(figsize=(10,10))
    plt.rcdefaults()
    for i in range(len(vals)):
        color = plt.cm.jet(i/(len(vals)))
        sns.distplot(avg_op[vals[i]][0], color=color, label = conc[i], kde_kws={'gridsize': 200, 'bw': 0.15} , kde=True, hist=False, norm_hist=True)
    plt.xticks(fontsize=20)
    plt.yticks( fontsize=20)
    plt.legend()
    #plt.axis([-1, 1, 0, 6])
    plt.xlabel("Vertex Curvature", fontsize=20)
    plt.ylabel("Average Density", fontsize=20)
    plt.title("$H_{2}O$ (Urea) - TIP4P", fontsize=20)
    plt.savefig("forman_avg_urea_vertex.pdf", dpi=200)
    plt.show()

    conc = ["1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M"]
    plt.figure(figsize=(10,10))
    plt.rcdefaults()
    for i in range(len(vals)):
        color = plt.cm.jet(i/(len(vals)))
        sns.distplot(avg_op[vals[i]][1], color=color, label = conc[i], kde_kws={'gridsize': 200, 'bw': 0.15} , kde=True, hist=False, norm_hist=True)
    plt.xticks(fontsize=20)
    plt.yticks( fontsize=20)
    plt.legend()
    #plt.axis([-1, 1, 0, 6])
    plt.xlabel("Edge Curvature", fontsize=20)
    plt.ylabel("Average Density", fontsize=20)
    plt.title("$H_{2}O$ (Urea) - TIP4P", fontsize=20)
    plt.savefig("forman_avg_urea_edge.pdf", dpi=200)
    plt.show()

    conc = ["1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M"]
    plt.figure(figsize=(10,10))
    plt.rcdefaults()
    for i in range(len(vals)):
        color = plt.cm.jet(i/(len(vals)))
        sns.distplot(avg_op[vals[i]][2], color=color, label = conc[i], kde_kws={'gridsize': 200, 'bw': 0.15} , kde=True, hist=False, norm_hist=True)
    plt.xticks(fontsize=20)
    plt.yticks( fontsize=20)
    plt.legend()
    #plt.axis([-1, 1, 0, 6])
    plt.xlabel("2-Simplex Curvature", fontsize=20)
    plt.ylabel("Average Density", fontsize=20)
    plt.title("$H_{2}O$ (Urea) - TIP4P", fontsize=20)
    #plt.savefig("forman_avg_urea_tri.pdf", dpi=200)
    plt.show()

def plot_urea_sample():
    vals = ["1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M"]
    urea_op = defaultdict(list)
    for v in vals:
        curv_0 = []
        curv_1 = []
        curv_2 = []
        data = np.load("forman_urea_raw_op_"+v+".npz", allow_pickle=True)
        for d in data.keys():
            curv_0.append(data[d][()][0])
            curv_1.append(data[d][()][1])
            curv_2.append(data[d][()][2])
        urea_op[v] = [curv_0, curv_1, curv_2]

    plt.figure()
    plt.rcParams.update({'font.size': 16})
    f, axes = plt.subplots(3,1, figsize=(10, 10), dpi=200, sharex=True)
    #plt.xlim(-1,1)
    plt.subplots_adjust(hspace=.3)
    sns.distplot(urea_op["8M"][0][-1], hist_kws=dict(edgecolor="k", lw=.5, alpha=0.7), color='tab:blue', kde_kws={"color":"tab:red", "lw": 2}, ax=axes[0], kde=False)
    axes[0].set(xlabel="Vertex Curvature", ylabel="No. of Vertices")
    sns.distplot(urea_op["8M"][1][-1], hist_kws=dict(edgecolor="k", lw=.5, alpha=0.7, width=.5), color='tab:blue', kde_kws={"color":"tab:red", "lw": 2}, ax=axes[1], kde=False)
    axes[1].set(xlabel="Edge Curvature", ylabel="No. of Edges")
    sns.distplot(urea_op["8M"][2][-1], hist_kws=dict(edgecolor="k", lw=.5, alpha=0.7, width=.5), color='tab:blue', kde_kws={"color":"tab:red", "lw": 2}, ax=axes[2], kde=False)

    axes[0].xaxis.set_tick_params(which='both', labelbottom=True)
    axes[1].xaxis.set_tick_params(which='both', labelbottom=True)
    axes[2].set(xlabel="2-Simplex Curvature", ylabel="No. of Triangles")
    #plt.xticks(fontsize=11)
    #plt.yticks( fontsize=11)
    #plt.legend()
    #plt.axis([-1, 1, 0, 6])
    #plt.xlabel("Vertex Curvature", fontsize=12)
    #plt.ylabel("Density", fontsize=12)
    #plt.title("$H_{2}O$ (Tmao) - TIP4P")
    plt.tight_layout()
    plt.savefig("forman_urea_8M_hist.pdf", dpi=200)
    plt.show()

    plt.figure()
    f, axes = plt.subplots(3,1, figsize=(10, 10), dpi=200, sharex=True)
    #plt.xlim(-1,1)
    plt.subplots_adjust(hspace=.3)
    sns.distplot(urea_op["8M"][0][-1], hist=False, hist_kws=dict(edgecolor="k", lw=.5, alpha=0.7), color='tab:blue', kde_kws={"color":"tab:blue", "lw": 2}, ax=axes[0], kde=True)
    axes[0].set(xlabel="Vertex Curvature", ylabel="Density")
    sns.distplot(urea_op["8M"][1][-1], hist=False, hist_kws=dict(edgecolor="k", lw=.5, alpha=0.7, width=.5), color='tab:blue', kde_kws={"color":"tab:blue", "lw": 2}, ax=axes[1], kde=True)
    axes[1].set(xlabel="Edge Curvature", ylabel="Density")
    sns.distplot(urea_op["8M"][2][-1], hist=False, hist_kws=dict(edgecolor="k", lw=.5, alpha=0.7, width=.5), color='tab:blue', kde_kws={"color":"tab:blue", "lw": 2}, ax=axes[2], kde=True)

    axes[0].xaxis.set_tick_params(which='both', labelbottom=True)
    axes[1].xaxis.set_tick_params(which='both', labelbottom=True)
    axes[2].set(xlabel="2-Simplex Curvature", ylabel="Density")
    #plt.xticks(fontsize=11)
    #plt.yticks( fontsize=11)
    #plt.legend()
    #plt.axis([-1, 1, 0, 6])
    #plt.xlabel("Vertex Curvature", fontsize=12)
    #plt.ylabel("Density", fontsize=12)
    #plt.title("$H_{2}O$ (Tmao) - TIP4P")
    plt.tight_layout()
    plt.savefig("forman_urea_8M_kde.pdf", dpi=200)
    plt.show()

def plot_tmao_sample():
    tmao_op = defaultdict(list)
    for v in vals:
        curv_0 = []
        curv_1 = []
        curv_2 = []
        data = np.load("forman_tmao_raw_op_"+v+".npz", allow_pickle=True)
        for d in data.keys():
            curv_0.append(data[d][()][0])
            curv_1.append(data[d][()][1])
            curv_2.append(data[d][()][2])
        tmao_op[v] = [curv_0, curv_1, curv_2]

    plt.figure()
    f, axes = plt.subplots(3,1, figsize=(10, 10), dpi=200, sharex=True)
    #plt.xlim(-1,1)
    plt.subplots_adjust(hspace=.3)
    sns.distplot(tmao_op["8M"][0][-1], hist_kws=dict(edgecolor="k", lw=.5, alpha=0.7), color='tab:blue', kde_kws={"color":"tab:red", "lw": 2}, ax=axes[0], kde=False)
    axes[0].set(xlabel="Vertex Curvature", ylabel="No. of Vertices")
    sns.distplot(tmao_op["8M"][1][-1], hist_kws=dict(edgecolor="k", lw=.5, alpha=0.7, width=.5), color='tab:blue', kde_kws={"color":"tab:red", "lw": 2}, ax=axes[1], kde=False)
    axes[1].set(xlabel="Edge Curvature", ylabel="No. of Edges")
    sns.distplot(tmao_op["8M"][2][-1], hist_kws=dict(edgecolor="k", lw=.5, alpha=0.7, width=.5), color='tab:blue', kde_kws={"color":"tab:red", "lw": 2}, ax=axes[2], kde=False)

    axes[0].xaxis.set_tick_params(which='both', labelbottom=True)
    axes[1].xaxis.set_tick_params(which='both', labelbottom=True)
    axes[2].set(xlabel="2-Simplex Curvature", ylabel="No. of Triangles")
    #plt.xticks(fontsize=11)
    #plt.yticks( fontsize=11)
    #plt.legend()
    #plt.axis([-1, 1, 0, 6])
    #plt.xlabel("Vertex Curvature", fontsize=12)
    #plt.ylabel("Density", fontsize=12)
    #plt.title("$H_{2}O$ (Tmao) - TIP4P")
    plt.tight_layout()
    plt.savefig("forman_tmao_8M_hist.pdf", dpi=200)
    plt.show()

    plt.figure()
    f, axes = plt.subplots(3,1, figsize=(10, 10), dpi=200, sharex=True)
    #plt.xlim(-1,1)
    plt.subplots_adjust(hspace=.3)
    sns.distplot(tmao_op["8M"][0][-1], hist=False, hist_kws=dict(edgecolor="k", lw=.5, alpha=0.7), color='tab:blue', kde_kws={"color":"tab:blue", "lw": 2}, ax=axes[0], kde=True)
    axes[0].set(xlabel="Vertex Curvature", ylabel="Density")
    sns.distplot(tmao_op["8M"][1][-1], hist=False, hist_kws=dict(edgecolor="k", lw=.5, alpha=0.7, width=.5), color='tab:blue', kde_kws={"color":"tab:blue", "lw": 2}, ax=axes[1], kde=True)
    axes[1].set(xlabel="Edge Curvature", ylabel="Density")
    sns.distplot(tmao_op["8M"][2][-1], hist=False, hist_kws=dict(edgecolor="k", lw=.5, alpha=0.7, width=.5), color='tab:blue', kde_kws={"color":"tab:blue", "lw": 2}, ax=axes[2], kde=True)

    axes[0].xaxis.set_tick_params(which='both', labelbottom=True)
    axes[1].xaxis.set_tick_params(which='both', labelbottom=True)
    axes[2].set(xlabel="2-Simplex Curvature", ylabel="Density")
    #plt.xticks(fontsize=11)
    #plt.yticks( fontsize=11)
    #plt.legend()
    #plt.axis([-1, 1, 0, 6])
    #plt.xlabel("Vertex Curvature", fontsize=12)
    #plt.ylabel("Density", fontsize=12)
    #plt.title("$H_{2}O$ (Tmao) - TIP4P")
    plt.tight_layout()
    plt.savefig("forman_tmao_8M_kde.pdf", dpi=200)
    plt.show()

def plot_tmao_median():
    vals = ["1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M"]
    tmao_op = defaultdict(list)
    for v in vals:
        curv_0 = []
        curv_1 = []
        curv_2 = []
        data = np.load("forman_tmao_raw_op_"+v+".npz", allow_pickle=True)
        for d in data.keys():
            curv_0.append(data[d][()][0])
            curv_1.append(data[d][()][1])
            curv_2.append(data[d][()][2])
        tmao_op[v] = [curv_0, curv_1, curv_2]

    points = []
    for i in range(len(vals)):
        plt.figure(figsize=(10,10), dpi=150)
        color = plt.cm.jet(i/(len(vals)))
        temp = []
        for j in range(101):
            ax = sns.kdeplot(tmao_op[vals[i]][0][j][0], bw=.15, gridsize=200)
            x = list(ax.lines[j].get_data()[0])
            y = list(ax.lines[j].get_data()[1])
            cdf = scipy.integrate.cumtrapz(y, x, initial=0)
            nearest_05 = np.abs(cdf-0.5).argmin()

            x_median = x[nearest_05]
            y_median = y[nearest_05]

            temp.append([x_median, y_median])
        points.append(temp)
    plt.show()
    points = np.array(points)

    vals = ["1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M"]
    plt.rcdefaults()
    plt.figure(figsize=(8,8))#, dpi=150)
    for i in range(len(vals)):
        color = plt.cm.jet(i/(len(vals)))
        for j in range(len(points[i])):
            plt.scatter(points[i][j][0], points[i][j][1], color=color)
        plt.plot([], [], label = vals[i], color=color)
    plt.xticks(fontsize=20)
    plt.yticks( fontsize=20)
    plt.legend()
    #plt.axis([-1, 1, 0, 6])
    plt.xlabel("Median Vertex Curvature", fontsize=20)
    plt.ylabel("Density", fontsize=20)
    plt.tight_layout()
    plt.savefig("forman_tmao_vertex_median.pdf", dpi=200)
    plt.show()

def plot_urea_median():
    vals = ["1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M"]
    urea_op = defaultdict(list)
    for v in vals:
        curv_0 = []
        curv_1 = []
        curv_2 = []
        data = np.load("forman_urea_raw_op_"+v+".npz", allow_pickle=True)
        for d in data.keys():
            curv_0.append(data[d][()][0])
            curv_1.append(data[d][()][1])
            curv_2.append(data[d][()][2])
        urea_op[v] = [curv_0, curv_1, curv_2]

    points = []
    for i in range(len(vals)):
        plt.figure(figsize=(10,10), dpi=150)
        color = plt.cm.jet(i/(len(vals)))
        temp = []
        for j in range(101):
            ax = sns.kdeplot(urea_op[vals[i]][0][j][0], bw=.15, gridsize=200)
            x = list(ax.lines[j].get_data()[0])
            y = list(ax.lines[j].get_data()[1])
            cdf = scipy.integrate.cumtrapz(y, x, initial=0)
            nearest_05 = np.abs(cdf-0.5).argmin()

            x_median = x[nearest_05]
            y_median = y[nearest_05]

            temp.append([x_median, y_median])
        points.append(temp)
    plt.show()
    points = np.array(points)

    plt.figure(figsize=(8,8))#, dpi=150)
    #_, cs = sns.kdeplot(np.reshape(points[:,:,0],(808,)), np.reshape(points[:,:,1],(808,)), cmap="jet")#, shade=True)
    #plt.clabel(cs, cs.levels, inline=True)
    for i in range(len(vals)):
        color = plt.cm.jet(i/(len(vals)))
        #sns.kdeplot(points[i,:,0], points[i,:,1], color=color, alpha=0.5)
        for j in range(len(points[i])):
            plt.scatter(points[i][j][0], points[i][j][1], color=color)
        plt.plot([], [], label = vals[i], color=color)
    plt.xticks(fontsize=20)
    plt.yticks( fontsize=20)
    plt.legend()
    #plt.axis([-1, 1, 0, 6])

    plt.xlabel("Median Vertex Curvature", fontsize=20)
    plt.ylabel("Density", fontsize=20)
    plt.tight_layout()
    plt.savefig("forman_urea_vertex_median.pdf", dpi=200)
    plt.show()

def plot_tmao_vertex_3d():
    vals = ["1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M"]

    tmao_op = defaultdict(list)
    for v in vals:
        curv_0 = []
        curv_1 = []
        curv_2 = []
        data = np.load("forman_tmao_raw_op_"+v+".npz", allow_pickle=True)
        for d in data.keys():
            curv_0.append(data[d][()][0])
            curv_1.append(data[d][()][1])
            curv_2.append(data[d][()][2])
        tmao_op[v] = [curv_0, curv_1, curv_2]

    for i in range(len(vals)):
        x = np.outer(range(0, 101), np.ones(256))
        y, z = [], []
        for j in range(101):
            ax = sns.kdeplot(tmao_op[vals[i]][0][j][0], bw=.15, gridsize=200)
            y.append(list(ax.lines[j].get_data()[0]))
            z.append(list(ax.lines[j].get_data()[1]))
        
        fig = plt.figure(figsize=(10,5), dpi=200)
        ax= fig.add_subplot(111, projection= '3d')
        ax.plot_trisurf(np.reshape(y, (256*101)), np.reshape(x, (256*101)), np.reshape(z, (256*101)), cmap="jet", vmin=0.25, vmax=0.32)
        #fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
        #fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                        #highlightcolor="limegreen", project_z=True))
        #surf = ax.plot_surface(np.array(y), np.array(x), np.array(z),cmap='jet',linewidth=0,antialiased='True',rstride=3,cstride=3)
        ax.contourf(np.array(y), np.array(x), np.array(z), 100, zdir='z', offset=-.2, cmap = "jet",  vmin=0.25, vmax=0.32)
        #ax.set_title('$H_{2}O$ (Tmao) - TIP4P - '+vals[i])
        ax.view_init(elev=10., azim=-120)
        ax.set_xlabel('Vertex Curvature', rotation=0, labelpad=15)
        ax.set_ylabel('Frame Index', rotation=0, labelpad=15)
        ax.set_zlabel('Density', rotation=90,labelpad=15)
        #ax.tick_params(axis='z', pad=10)
        ax.zaxis.set_rotate_label(False)
        ax.grid(False)
        #ax.set_xlim([-12, 4])
        #ax.set_ylim([0, 100])
        ax.set_zlim([-.2, 0.4])
        m = plt.cm.ScalarMappable(cmap=plt.cm.jet)
        m.set_clim(0.25, 0.32)
        plt.colorbar(m, extend="both", shrink=0.75, pad=-0.05)
        plt.tight_layout()
        plt.savefig("forman_tmao_3d_vertex_"+vals[i]+".pdf", dpi=200)
        #fig.colorbar(surf)
        plt.show()

def plot_urea_vertex_3d():
    vals = ["1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M"]
    urea_op = defaultdict(list)
    for v in vals:
        curv_0 = []
        curv_1 = []
        curv_2 = []
        data = np.load("forman_urea_raw_op_"+v+".npz", allow_pickle=True)
        for d in data.keys():
            curv_0.append(data[d][()][0])
            curv_1.append(data[d][()][1])
            curv_2.append(data[d][()][2])
        urea_op[v] = [curv_0, curv_1, curv_2]

    for i in range(len(vals)):
        x = np.outer(range(0, 101), np.ones(256))
        y, z = [], []
        for j in range(101):
            ax = sns.kdeplot(urea_op[vals[i]][0][j][0], bw=.15, gridsize=200)
            y.append(list(ax.lines[j].get_data()[0]))
            z.append(list(ax.lines[j].get_data()[1]))
        
        fig = plt.figure(figsize=(10,5), dpi=200)
        ax= fig.add_subplot(111, projection= '3d')
        ax.plot_trisurf(np.reshape(y, (256*101)), np.reshape(x, (256*101)), np.reshape(z, (256*101)), cmap="jet", vmin=0.2, vmax=0.27)
        #fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
        #fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                        #highlightcolor="limegreen", project_z=True))
        #surf = ax.plot_surface(np.array(y), np.array(x), np.array(z),cmap='jet',linewidth=0,antialiased='True',rstride=3,cstride=3)
        ax.contourf(np.array(y), np.array(x), np.array(z), 100, zdir='z', offset=-.2, cmap = "jet",  vmin=0.2, vmax=0.27)
        #ax.set_title('$H_{2}O$ (Tmao) - TIP4P - '+vals[i])
        ax.view_init(elev=10., azim=-120)
        ax.set_xlabel('Vertex Curvature', rotation=0, labelpad=15)
        ax.set_ylabel('Frame Index', rotation=0, labelpad=15)
        ax.set_zlabel('Density', rotation=90,labelpad=15)
        #ax.tick_params(axis='z', pad=10)
        ax.zaxis.set_rotate_label(False)
        ax.grid(False)
        #ax.set_xlim([-12, 4])
        #ax.set_ylim([0, 100])
        ax.set_zlim([-.2, 0.4])
        m = plt.cm.ScalarMappable(cmap=plt.cm.jet)
        m.set_clim(0.2, 0.27)
        plt.colorbar(m, extend="both", shrink=0.75, pad=-0.05)
        plt.tight_layout()
        plt.savefig("forman_urea_3d_vertex_"+vals[i]+".pdf", dpi=200)
        #fig.colorbar(surf)
        plt.show()