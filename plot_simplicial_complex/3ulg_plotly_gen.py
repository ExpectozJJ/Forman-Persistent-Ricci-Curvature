from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GeneralisedFormanRicci.frc import GeneralisedFormanRicci, gen_graph, n_faces
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import math
import matplotlib as mpl
import matplotlib
import plotly.io as pio
import matplotlib.pyplot as plt

pio.orca.config.use_xvfb = True

def rotate_z(x, y, z, theta):
    w = x+1j*y
    return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale

""" Normalise Colormap to unique range """ 
seismic_cmap = matplotlib.cm.get_cmap('seismic')

seismic_rgb = []
norm = mpl.colors.Normalize(vmin=0, vmax=255)

for i in range(0, 255):
    k = mpl.colors.colorConverter.to_rgb(seismic_cmap(norm(i)))
    seismic_rgb.append(k)

seismic = matplotlib_to_plotly(seismic_cmap, 255)


""" Generate ORC and FRC for PDBID: 3ULG """
for chain in ["chainA", "chainB", "chainC"]:
    print("Running: ", chain)
    
    data = np.load("3ulg_CA_"+chain+".npz", allow_pickle=True)
    for d in data["PRO"]:
        data = d["pos"]
    
    sc = GeneralisedFormanRicci(data, epsilon = 15)
    G = gen_graph(list(n_faces(sc.S, 1)), sc.pts, sc.labels)
    ans = sc.compute_forman()
    node_dict = ans[0]
    edge_dict = ans[1]
    tri_dict = ans[2]

    #np.save("3ulg_simplices_15.npy", sc.S)
    np.save("3ulg_CA_"+chain+"_15.npy", ans)
    nx.write_gpickle(G, "3ulg_CA_"+chain+"_15.gpickle")
    
    orc = OllivierRicci(G, alpha=0.5)
    orc.compute_ricci_curvature()

    orc_node = nx.get_node_attributes(orc.G, "ricciCurvature")
    orc_edge = nx.get_edge_attributes(orc.G, "ricciCurvature")

    np.save("3ulg_CA_"+chain+"_ollivier_15.npy", [orc_node, orc_edge])
    nx.write_gpickle(orc.G, "3ulg_CA_"+chain+"_ollivier_15.gpickle")

"""                   Plot 0-Simplex FRC                       """

edge_x = []
edge_y = []
edge_z = []
traces = []
node_x = []
node_y = []
node_z = []

node_frc = []

for chain in ["chainA", "chainB", "chainC"]:
    G = nx.read_gpickle("3ulg_CA_"+chain+"_15.gpickle")
    ans = np.load("3ulg_CA_"+chain+"_15.npy", allow_pickle=True)
    #ans = sc.compute_forman()
    #node_dict = ans[0]
    #edge_dict = ans[1]
    #tri_dict = ans[2]

    node_dict = ans[()][0]
    edge_dict = ans[()][1]
    tri_dict = ans[()][2]


    node_frc += list(node_dict.values())
    for edge in G.edges():
        x0, y0, z0 = G.nodes[edge[0]]['coords']
        x1, y1, z1 = G.nodes[edge[1]]['coords']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_z.append(z0)
        edge_z.append(z1)
        edge_z.append(None)

    for node in G.nodes():
        x, y, z = G.nodes[node]['coords']
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        
    """ Shade 2-Simplices """
    """
    for key, val in tri_dict.items():
        s = G.nodes[key[0]]['coords']
        t = G.nodes[key[1]]['coords']
        u = G.nodes[key[2]]['coords']
        e = tri_dict[key]
        #color = mpl.cm.coolwarm(norm(e), bytes=True)
        #tri_c[key] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])
        traces.append(go.Mesh3d(x=[s[0], t[0], u[0]], y=[s[1], t[1], u[1]], z=[s[2], t[2], u[2]], color='lightgray', opacity=.01))
    """
        
traces.append(go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    line=dict(width=3, color='gray'),
    hoverinfo='none', opacity=.2,
    mode='lines'))

traces.append(go.Scatter3d(
    x=node_x, y=node_y, z=node_z,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        colorscale=seismic, color = node_frc, cmin=-10, cmax=20,
        size=10, opacity=1, line=dict(color='black', width=3), colorbar=dict(thickness=20, outlinewidth=2, tickfont=dict(family='Arial', size=30))
        ),
        line_width=2))


fig = go.Figure(data=traces,
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')

fig.update_layout(width=1000, height=1000, scene = dict(
    xaxis=dict(
    autorange=True,
    showgrid=False,
    zeroline=False,
    showline=False,
    showbackground=False,
    title='',
    ticks='',
    showticklabels=False
),
yaxis=dict(
    autorange=True,
    showgrid=False,
    zeroline=False,
    showline=False,
    showbackground=False,
    ticks='',
    title='',
    showticklabels=False
),
zaxis=dict(
    autorange=True,
    showgrid=False,
    zeroline=False,
    showline=False,
    showbackground=False,
    title='',
    ticks='',
    showticklabels=False
)))

xe, ye, ze = rotate_z(0, 0, 1.75, -.1)

camera = dict(
    up=dict(x=0, y=1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=xe, y=ye, z=ze)
)

fig.update_layout(scene_camera=camera, scene_dragmode='orbit')
fig.write_html("3ulg_CA_forman_vertex_15.html")
#fig.write_image("rect_"+str(f)+"_vertex.png", scale=2)
#fig.show()

"""                   Plot 1-Simplex FRC                       """

a, b = -20, 20
norm = mpl.colors.Normalize(vmin=a, vmax=b)

traces = []
node_x = []
node_y = []
node_z = []

edge_frc = []

for chain in ["chainA", "chainB", "chainC"]:
    G = nx.read_gpickle("3ulg_CA_"+chain+"_15.gpickle")
    ans = np.load("3ulg_CA_"+chain+"_15.npy", allow_pickle=True)
    #ans = sc.compute_forman()
    #node_dict = ans[0]
    #edge_dict = ans[1]
    #tri_dict = ans[2]

    node_dict = ans[()][0]
    edge_dict = ans[()][1]
    tri_dict = ans[()][2]
    
    edge_c = dict()

    edge_frc += list(edge_dict.values())
    for edge in G.edges():
        edge_x = []
        edge_y = []
        edge_z = []
        x0, y0, z0 = G.nodes[edge[0]]['coords']
        x1, y1, z1 = G.nodes[edge[1]]['coords']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_z.append(z0)
        edge_z.append(z1)
        edge_z.append(None)
        
        e = edge_dict[edge]
        #e = ((e-np.mean(list(edge_dict.values())))/(np.std(list(edge_dict.values()))))
        #rint(norm(e))
        color = mpl.cm.seismic(norm(e), bytes=True)
        edge_c[edge] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])

        traces.append(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=10, color=edge_c[edge]),
            hoverinfo='none', opacity=.25,
            mode='lines'))

    for node in G.nodes():
        x, y, z = G.nodes[node]['coords']
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        
    """ Shade 2-Simplices """
    """
    for key, val in tri_dict.items():
        s = G.nodes[key[0]]['coords']
        t = G.nodes[key[1]]['coords']
        u = G.nodes[key[2]]['coords']
        e = tri_dict[key]
        #color = mpl.cm.coolwarm(norm(e), bytes=True)
        #tri_c[key] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])
        traces.append(go.Mesh3d(x=[s[0], t[0], u[0]], y=[s[1], t[1], u[1]], z=[s[2], t[2], u[2]], color='lightgray', opacity=.05))
    """

node_trace = go.Scatter3d(
x=node_x, y=node_y, z=node_z,
mode='markers',
hoverinfo='text',
marker=dict(
    color = 'black',
    size=2, opacity=1, line=dict(color='black', width=1)
    ),
    line_width=2)

traces.append(node_trace)
    
color_range = seismic_cmap
color_trace = go.Scatter3d(x=node_x,y=node_y, z=node_z,
                         mode='markers',
                         marker=go.scatter3d.Marker(colorscale=seismic, cmin=a, cmax=b,
                            size=.01,
                            color=edge_frc,
                            showscale=True, colorbar=dict(thickness=20, outlinewidth=2, tickfont=dict(family='Arial', size=30))
                            )
                        )
traces.append(color_trace)


fig = go.Figure(data=traces,
             layout=go.Layout(
                showlegend=False, 
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')

fig.update_layout(width = 1000, height = 1000, scene = dict(
    xaxis=dict(
    autorange=True,
    showgrid=False,
    zeroline=False,
    showline=False,
    showbackground=False,
    title='',
    ticks='',
    showticklabels=False
),
yaxis=dict(
    autorange=True,
    showgrid=False,
    zeroline=False,
    showline=False,
    showbackground=False,
    ticks='',
    title='',
    showticklabels=False
),
zaxis=dict(
    autorange=True,
    showgrid=False,
    zeroline=False,
    showline=False,
    showbackground=False,
    title='',
    ticks='',
    showticklabels=False
)))

xe, ye, ze = rotate_z(0, 0, 1.75, -.1)

camera = dict(
    up=dict(x=0, y=1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=xe, y=ye, z=ze)
)

fig.update_layout(scene_camera=camera, scene_dragmode='orbit')
fig.write_html("3ulg_CA_forman_edge_15.html")
#fig.write_image("rect_"+str(f)+"_edge.png", scale=2)
#fig.show()

"""                Plot 2-Simplex FRC                         """

edge_x = []
edge_y = []
edge_z = []
traces = []
node_x = []
node_y = []
node_z = []

tri_frc = []

a, b = -20, 20
norm = mpl.colors.Normalize(vmin=a, vmax=b)

for chain in ["chainA", "chainB", "chainC"]:
    G = nx.read_gpickle("3ulg_CA_"+chain+"_15.gpickle")
    ans = np.load("3ulg_CA_"+chain+"_15.npy", allow_pickle=True)
    #ans = sc.compute_forman()
    #node_dict = ans[0]
    #edge_dict = ans[1]
    #tri_dict = ans[2]

    node_dict = ans[()][0]
    edge_dict = ans[()][1]
    tri_dict = ans[()][2]

    tri_c = dict()
    tri_frc += list(tri_dict.values())
    for edge in G.edges():
        x0, y0, z0 = G.nodes[edge[0]]['coords']
        x1, y1, z1 = G.nodes[edge[1]]['coords']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_z.append(z0)
        edge_z.append(z1)
        edge_z.append(None)

    for node in G.nodes():
        x, y, z = G.nodes[node]['coords']
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        
    for key, val in tri_dict.items():
        s = G.nodes[key[0]]['coords']
        t = G.nodes[key[1]]['coords']
        u = G.nodes[key[2]]['coords']
        e = tri_dict[key]
        color = mpl.cm.seismic(norm(e), bytes=True)
        tri_c[key] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])
        traces.append(go.Mesh3d(x=[s[0], t[0], u[0]], y=[s[1], t[1], u[1]], z=[s[2], t[2], u[2]], color=tri_c[key], opacity=.5))
        
traces.append(go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    line=dict(width=3, color='gray'),
    hoverinfo='none', opacity=.2,
    mode='lines'))

node_trace = go.Scatter3d(
x=node_x, y=node_y, z=node_z,
mode='markers',
hoverinfo='text',
marker=dict(
    color = 'black',
    size=2, opacity=1, line=dict(color='black', width=1)
    ),
    line_width=2)

traces.append(node_trace)

color_range = seismic_cmap
color_trace = go.Scatter3d(x=node_x,y=node_y, z=node_z,
                         mode='markers',
                         marker=go.scatter3d.Marker(colorscale=seismic, cmin=a, cmax=b,
                            size=.01,
                            color=tri_frc,
                            showscale=True, colorbar=dict(thickness=20, outlinewidth=2, tickfont=dict(family='Arial', size=30))
                            )
                        )
traces.append(color_trace)


fig = go.Figure(data=traces,
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')

fig.update_layout(width=1000, height=1000, scene = dict(
    xaxis=dict(
    autorange=True,
    showgrid=False,
    zeroline=False,
    showline=False,
    showbackground=False,
    title='',
    ticks='',
    showticklabels=False
),
yaxis=dict(
    autorange=True,
    showgrid=False,
    zeroline=False,
    showline=False,
    showbackground=False,
    ticks='',
    title='',
    showticklabels=False
),
zaxis=dict(
    autorange=True,
    showgrid=False,
    zeroline=False,
    showline=False,
    showbackground=False,
    title='',
    ticks='',
    showticklabels=False
)))

xe, ye, ze = rotate_z(0, 0, 1.75, -.1)

camera = dict(
    up=dict(x=0, y=1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=xe, y=ye, z=ze)
)

fig.update_layout(scene_camera=camera, scene_dragmode='orbit')
fig.write_html("3ulg_CA_forman_triangle_15.html")
#fig.write_image("rect_"+str(f)+"_vertex.png", scale=2)
#fig.show()

"""               Generate Vertex ORC                      """

edge_x = []
edge_y = []
edge_z = []
traces = []
node_x = []
node_y = []
node_z = []

node_frc = []

for chain in ["chainA", "chainB", "chainC"]:
    G = nx.read_gpickle("3ulg_CA_"+chain+"_ollivier_15.gpickle")
    ans = np.load("3ulg_CA_"+chain+"_ollivier_15.npy", allow_pickle=True)
    node_dict, edge_dict= ans[0], ans[1]

    node_frc += list(node_dict.values())
    for edge in G.edges():
        x0, y0, z0 = G.nodes[edge[0]]['coords']
        x1, y1, z1 = G.nodes[edge[1]]['coords']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_z.append(z0)
        edge_z.append(z1)
        edge_z.append(None)

    for node in G.nodes():
        x, y, z = G.nodes[node]['coords']
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        
traces.append(go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    line=dict(width=3, color='gray'),
    hoverinfo='none', opacity=.2,
    mode='lines'))

traces.append(go.Scatter3d(
    x=node_x, y=node_y, z=node_z,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        colorscale=seismic, color = node_frc, cmin=0, cmax=0.5,
        size=10, opacity=1, line=dict(color='black', width=3), colorbar=dict(thickness=20, outlinewidth=2, tickfont=dict(family='Arial', size=30))
        ),
        line_width=2))


fig = go.Figure(data=traces,
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')

fig.update_layout(width=1000, height=1000, scene = dict(
    xaxis=dict(
    autorange=True,
    showgrid=False,
    zeroline=False,
    showline=False,
    showbackground=False,
    title='',
    ticks='',
    showticklabels=False
),
yaxis=dict(
    autorange=True,
    showgrid=False,
    zeroline=False,
    showline=False,
    showbackground=False,
    ticks='',
    title='',
    showticklabels=False
),
zaxis=dict(
    autorange=True,
    showgrid=False,
    zeroline=False,
    showline=False,
    showbackground=False,
    title='',
    ticks='',
    showticklabels=False
)))

xe, ye, ze = rotate_z(0, 0, 1.75, -.1)

camera = dict(
    up=dict(x=0, y=1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=xe, y=ye, z=ze)
)

fig.update_layout(scene_camera=camera, scene_dragmode='orbit')
fig.write_html("3ulg_CA_ollivier_vertex_15.html")
#fig.write_image("rect_"+str(f)+"_vertex.png", scale=2)
#fig.show()

"""        Generate Edge ORC          """
a, b = -.4, .4
norm = mpl.colors.Normalize(vmin=a, vmax=b)


traces = []
node_x = []
node_y = []
node_z = []

edge_frc = []

for chain in ["chainA", "chainB", "chainC"]:
    G = nx.read_gpickle("3ulg_CA_"+chain+"_ollivier_15.gpickle")
    ans = np.load("3ulg_CA_"+chain+"_ollivier_15.npy", allow_pickle=True)
    node_dict, edge_dict= ans[0], ans[1]
    
    edge_c = dict()

    edge_frc += list(edge_dict.values())
    for edge in G.edges():
        edge_x = []
        edge_y = []
        edge_z = []
        x0, y0, z0 = G.nodes[edge[0]]['coords']
        x1, y1, z1 = G.nodes[edge[1]]['coords']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_z.append(z0)
        edge_z.append(z1)
        edge_z.append(None)
        
        e = edge_dict[edge]
        #e = ((e-np.mean(list(edge_dict.values())))/(np.std(list(edge_dict.values()))))
        #rint(norm(e))
        color = mpl.cm.seismic(norm(e), bytes=True)
        edge_c[edge] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])

        traces.append(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=10, color=edge_c[edge]),
            hoverinfo='none', opacity=.25,
            mode='lines'))

    for node in G.nodes():
        x, y, z = G.nodes[node]['coords']
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        
    """
    for key, val in tri_dict.items():
        s = G.nodes[key[0]]['coords']
        t = G.nodes[key[1]]['coords']
        u = G.nodes[key[2]]['coords']
        e = tri_dict[key]
        #color = mpl.cm.coolwarm(norm(e), bytes=True)
        #tri_c[key] = "rgba({},{},{},{})".format(color[0],color[1],color[2], color[3])
        traces.append(go.Mesh3d(x=[s[0], t[0], u[0]], y=[s[1], t[1], u[1]], z=[s[2], t[2], u[2]], color='lightgray', opacity=.05))
    """

node_trace = go.Scatter3d(
x=node_x, y=node_y, z=node_z,
mode='markers',
hoverinfo='text',
marker=dict(
    color = 'black',
    size=2, opacity=1, line=dict(color='black', width=1)
    ),
    line_width=2)

traces.append(node_trace)
    
color_range = seismic_cmap
color_trace = go.Scatter3d(x=node_x,y=node_y, z=node_z,
                         mode='markers',
                         marker=go.scatter3d.Marker(colorscale=seismic, cmin=a, cmax=b,
                            size=.01,
                            color=edge_frc,
                            showscale=True, colorbar=dict(thickness=20, outlinewidth=2, tickfont=dict(family='Arial', size=30))
                            )
                        )
traces.append(color_trace)


fig = go.Figure(data=traces,
             layout=go.Layout(
                showlegend=False, 
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')

fig.update_layout(width = 1000, height = 1000, scene = dict(
    xaxis=dict(
    autorange=True,
    showgrid=False,
    zeroline=False,
    showline=False,
    showbackground=False,
    title='',
    ticks='',
    showticklabels=False
),
yaxis=dict(
    autorange=True,
    showgrid=False,
    zeroline=False,
    showline=False,
    showbackground=False,
    ticks='',
    title='',
    showticklabels=False
),
zaxis=dict(
    autorange=True,
    showgrid=False,
    zeroline=False,
    showline=False,
    showbackground=False,
    title='',
    ticks='',
    showticklabels=False
)))

xe, ye, ze = rotate_z(0, 0, 1.75, -.1)

camera = dict(
    up=dict(x=0, y=1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=xe, y=ye, z=ze)
)

fig.update_layout(scene_camera=camera, scene_dragmode='orbit')
fig.write_html("3ulg_CA_ollivier_edge_15.html")
#fig.write_image("rect_"+str(f)+"_edge.png", scale=2)
#fig.show()
