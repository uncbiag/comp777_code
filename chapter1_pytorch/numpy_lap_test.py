from numpy import linalg as LA
import numpy as np
import pandas as pd
import hvplot.pandas
from bokeh.plotting import show
import holoviews as hv
from holoviews import opts

def eigen(A):
    # compute sorted egenvalues and eigenvectors
    eigen_values, eigen_vectors = LA.eig(A)
    idx = np.argsort(eigen_values)
    idx = idx[::-1] # reverse (so that is from large to small)
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:,idx]
    return (eigen_values, eigen_vectors)

def plot_evs(v,title='eigenvectors'):
    # plot eigenvectors
    shape = v.shape
    nr_pts = shape[1]
    nr_vecs = shape[0]
    group = np.repeat(list(range(nr_vecs)),nr_pts)
    x = list(range(nr_pts))*nr_vecs
    y = []
    for i in range(nr_vecs):
        y = y + list(v[:,i])
    df = pd.DataFrame({'group': group, 'x': x, 'y': y})
    ev_plot = df.hvplot('x','y',by='group',width=1500,height=1000, title=title)
    return ev_plot

def plot_iter(M,v0,dt,iter=10,subiter=1000000,title='iterations'):
    # plot iterations
    nr_pts = v0.shape[0]
    group = np.repeat(list(range(iter+1)), nr_pts)
    x = list(range(nr_pts)) * (iter+1)
    y = list(v0)
    v = v0
    for i in range(iter):
        print('Iter: {}/{}'.format(i+1,iter))
        for j in range(subiter):
            v = v-dt*M.dot(v)
        y = y + list(v)
    df = pd.DataFrame({'group': group, 'x': x, 'y': y})
    iter_plot = df.hvplot('x', 'y', by='group', width=1500, height=1000, title=title)
    return iter_plot

def create_A_neumann(nr):
    A_full = np.zeros([nr,nr])
    for i in range(nr-1):
        A_full[i,i] = -1
        A_full[i,i+1] = 1
    return A_full

# assume 10 points in the domain
# assume zero Neumann boundary conditions
# assume we are approximating derivatives based on right-sided derivatives

# A_full = np.array([[-1,1,0,0,0,0,0,0,0,0],
#                   [0,-1,1,0,0,0,0,0,0,0],
#                   [0,0,-1,1,0,0,0,0,0,0],
#                   [0,0,0,-1,1,0,0,0,0,0],
#                   [0,0,0,0,-1,1,0,0,0,0],
#                   [0,0,0,0,0,-1,1,0,0,0],
#                   [0,0,0,0,0,0,-1,1,0,0],
#                   [0,0,0,0,0,0,0,-1,1,0],
#                   [0,0,0,0,0,0,0,0,-1,1],
#                   [0,0,0,0,0,0,0,0,0,0]])

nr_of_points = 25

A_full = create_A_neumann(nr_of_points)/nr_of_points

# create an initial profile
v0 = np.zeros([nr_of_points,1])
v0[nr_of_points//4:3*nr_of_points//4,0] = 1

# now create the Laplacian matrix
L = (A_full.T).dot(A_full)
w,v = eigen(L)

# plot eigenvectors (only if there are relatively few)
if nr_of_points<=20:
    plt_v = plot_evs(v,title='eigenvectors L')
    show(hv.render(plt_v))

# and create the Laplacian squared

Lsqr = (L.T).dot(L)

ws,vs = eigen(Lsqr)

# plot eigenvectors (only if there are relatively few)
if nr_of_points<=20:
    plt_vs = plot_evs(vs,title='eigenvectors Lsqr')
    show(hv.render(plt_vs))

dt_lsqr = 0.75/ws[0]
plt_iter_lsqr = plot_iter(Lsqr,v0,dt=dt_lsqr,iter=20,subiter=1000,title='iterations Lsqr')
show(hv.render(plt_iter_lsqr))

dt_l = 0.75/w[0]
plt_iter_l = plot_iter(Lsqr,v0,dt=dt_l,iter=20,subiter=100000,title='iterations for L')
show(hv.render(plt_iter_l))