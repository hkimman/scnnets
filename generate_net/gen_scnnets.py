#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:50:22 2020

@author: ccd_com
"""

import numpy as np
from itertools import product
import networkx as nx
from scipy import stats
from scipy.sparse import csr_matrix
import cv2
from scipy.stats import multivariate_normal as mn
from tqdm import tqdm
from multiprocessing import Pool
from contextlib import closing
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import os
import argparse
from itertools import chain
from numba import jit
pca = PCA()

# =============================================================================
# loading template image from back_G file. The blue and red areas of back_G.png indicate
# core and shell, respectively.
# If you want to change the template image, just modify the below line.
# =============================================================================
tmp_Map = cv2.imread('/home/kh/문서/Codes/Python/SCN_modeling/back_G.png')
resize_scale = 0.0546
Map = np.round(cv2.resize(tmp_Map,None,fx=resize_scale,fy=resize_scale,interpolation=cv2.INTER_CUBIC)/255)
P_size = Map[:,:,2].shape
Core_ind = np.argwhere(Map[:,:,2]);
Shell_ind = np.argwhere(Map[:,:,0]);
All_ind = np.append(Core_ind,Shell_ind,axis=0)
core_center = Core_ind.mean(axis=0)
Except_ind = np.argwhere(np.logical_not(Map[:,:,0]+Map[:,:,2]));
N_cell = len(All_ind);
tmp_ED_M = [np.linalg.norm(All_ind - s,axis=1) for s in All_ind]
ED_M = np.matrix(tmp_ED_M)
val_Area = 420000*0.85
sc_len = np.sqrt(val_Area/N_cell)
sc_d = ED_M*sc_len;
Pos_mat = np.zeros((N_cell,N_cell))
P_y = np.max(All_ind[:,0]) - All_ind[:,0]
P_x = All_ind[:,1]
Pos_mat = np.c_[P_x,P_y]

def scale_mat_col(X, x_min=0, x_max=1):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 


def get_c(G):
    tmp_a = nx.clustering(G)
    return np.fromiter(tmp_a.values(),dtype=float).mean()
def get_r(G):
    tmp_b_1 = nx.degree_assortativity_coefficient(G,'in','in')
    tmp_b_2 = nx.degree_assortativity_coefficient(G,'out','out')
    tmp_b_3 = nx.degree_assortativity_coefficient(G,'in','out')
    tmp_b_4 = nx.degree_assortativity_coefficient(G,'out','in')
    test_b = np.array([tmp_b_1,tmp_b_2,tmp_b_3,tmp_b_4])
    return test_b

def beta_fit(a):
    return np.array(stats.beta.fit(a))

    
triu_idx = np.triu_indices(N_cell,1)
triu_idx_r = np.ravel_multi_index(triu_idx, (N_cell,)*2)
tril_idx = np.tril_indices(N_cell,-1)
tril_idx_r = np.ravel_multi_index(tril_idx, (N_cell,)*2)

@jit()
def tan_prob(arr,c1=5,c2=2.5):
    out = (1-np.tanh((arr-c1)/c2))/2
    return out

def norm_arr(arr):
    return (arr-arr.min())/arr.max()


def get_prob(in_deg,out_deg,ac):
    deg_diff_i = np.abs(in_deg - in_deg[:,np.newaxis])[triu_idx]
    deg_diff_o = np.abs(out_deg - out_deg[:,np.newaxis])[triu_idx]
    deg_diff_io = np.abs(out_deg - in_deg[:,np.newaxis])
    
    scale_i = norm_arr(deg_diff_i)
    scale_o = norm_arr(deg_diff_o)
    scale_io = norm_arr(deg_diff_io)
    all_sdrf = np.vstack((scale_i,scale_o,scale_io[triu_idx],scale_io.T[triu_idx])).T
    prop_arrs = np.zeros(all_sdrf.shape)
    for i in range(4):
        if ac[i] > 0:
            c1 = np.percentile(all_sdrf[:,i],5)
            c2 = 0.05
            prop_arrs[:,i] = tan_prob(all_sdrf[:,i],c1,c2)
        elif ac[i] < 0:
            c1 = np.percentile(1-all_sdrf[:,i],5)
            c2 = 0.05
            prop_arrs[:,i] = tan_prob(1-all_sdrf[:,i],c1,c2)
    totp = prop_arrs.sum(axis=0)
    rpd = np.divide(totp.max(), totp, where=totp!=0)
    p_d = np.zeros((N_cell,N_cell))
    p_d[triu_idx] = (prop_arrs*np.abs(ac)*rpd).sum(axis=1)
    return p_d + p_d.T

if not(N_cell%2):
    CD_M_ele = np.cumsum(np.concatenate((np.array([0]),np.ones(int(N_cell/2-1))*2,np.array([1,-1]),-np.ones(int(N_cell/2-2))*2)))
else:
    CD_M_ele = np.cumsum(np.concatenate((np.array([0]),np.ones(int((N_cell-1)/2))*2,np.array([0]),-np.ones(int((N_cell-1)/2)-1)*2)))
CD_M = np.zeros((N_cell,N_cell))
for i in range(N_cell):
    CD_M[:,i] = np.roll(CD_M_ele,i)
CD_S = np.tril(CD_M)
CD_S[CD_S ==0]=1e+5
P_M_el = norm_arr(CD_M[triu_idx])
P_M = np.zeros((N_cell,N_cell))
P_M[triu_idx] = P_M_el
P_M = P_M + P_M.T

def make_net(c_arr=[1,0,0],ac=[1,0,0,0],prop='prop',\
             prop_scale=20,SPD_opt1='p',SPD_opt2='tot',SPD_opt3=0):
    c_w,a_w,r_w = c_arr
    n_in = np.random.exponential(8.9,size=N_cell).astype(int) #assign the indegress based on experimental results
    non_in = n_in==0
    if prop == 'prop': # setting in&out degree relation
        probs_data = stats.beta.cdf(n_in,a=1,b=1,scale=prop_scale) + 1e-15
    elif prop == 'inv':
        probs_data = 1-stats.beta.cdf(n_in,a=5,b=20,scale=prop_scale)
    elif prop == 'none':
        probs_data = np.ones(N_cell)
    test_M = np.zeros((N_cell,N_cell))
    out_degs = np.zeros(N_cell)
    mnrv = mn(core_center,[[400, 0], [0, 400]])
    
    # draft network with in&out degree relation
    for i in range(N_cell):
        n_of_conn = n_in[i];
        dummy_ind = np.delete(np.arange(0,N_cell),i)
        dummy_ind = np.setdiff1d(dummy_ind,np.argwhere(test_M[i,:]!=0),True)
        non_out = np.logical_not(out_degs.astype(bool))
        isolated_nodes = np.argwhere(np.logical_and(non_in,non_out)).ravel()        
        must_contain_ind = np.intersect1d(dummy_ind, isolated_nodes,True)
        T_probs_data = probs_data.copy()
        T_probs_data[must_contain_ind] = 1
        probs = T_probs_data[dummy_ind]
        tmp_ind = np.random.choice(dummy_ind,n_of_conn,p=probs/probs.sum(),replace=False)
        test_M[tmp_ind,i] += 1
        out_degs[tmp_ind] += 1
    n_out = test_M.sum(axis=1)
    sn_in = n_in
    sn_out = n_out
    
    # generating probability matrix for rewiring
    p_p = tan_prob(P_M,0.003,0.001) # probability matrix for clustering coefficient
    cutoff_p = p_p <=0.5e-15
    p_p[cutoff_p] = 0
    p_d = get_prob(sn_in,sn_out,ac) # probability matrix for assortativity coefficient
    p_r = np.random.rand(p_p.shape[0],p_p.shape[1]) # random probability matrix 
    p_a = c_w*500*p_p + a_w*((1-r_w)*p_d + r_w*p_r)
    if np.nonzero(p_a)[0].size == 0:
        p_a = p_r
        p_a[p_a==0]=p_a[p_a!=0].min()
    else:
        p_a[p_a==0]=p_a[p_a!=0].min()
        
    # rewiring using probability matrix
    Lat_M = np.zeros((N_cell,N_cell))
    Lat_M_out = np.zeros(N_cell)
    N_cell_rand = np.arange(N_cell).astype(int)
    np.random.shuffle(N_cell_rand)
    for i in N_cell_rand:
        test_d = np.delete(np.arange(N_cell),i)
        test_d = test_d[Lat_M[i,test_d]==0]
        for_space_out = np.argwhere(sn_out > Lat_M_out).ravel()
        in_candidate =  np.intersect1d(for_space_out,test_d,True)
        if sn_in[i] < in_candidate.size:
            c_ind = np.random.choice(in_candidate,p=p_a[in_candidate,i]/\
                                     p_a[in_candidate,i].sum(),size=sn_in[i],replace=False)
        else:
            c_ind = in_candidate
        Lat_M[c_ind,i] += 1
        Lat_M_out[c_ind] += 1
    
    # preventing a isolated node
    tmp_G = nx.from_numpy_array(Lat_M)
    C_nn = np.array([np.array(list(s)) for s in nx.connected_components(tmp_G)],dtype=object)
    C_nn_size = np.array([s.size for s in C_nn])
    n_C_nn = len(C_nn)
    if n_C_nn != 1:
        G_c_arg = C_nn_size.argmax()
        R_c_arg = np.setdiff1d(np.arange(n_C_nn), G_c_arg, True)
        
        small_sel = np.empty(n_C_nn-1,dtype=int)
        for i,v in enumerate(C_nn[R_c_arg]):
            small_sel[i] = np.random.choice(v)
        big_sel = np.empty(small_sel.shape,dtype=int)
        buck_list = C_nn[G_c_arg].copy()
        
        p_a_lat = get_prob(Lat_M.sum(axis=0),Lat_M.sum(axis=1),ac)
        if np.nonzero(p_a_lat)[0].size == 0:
            p_a_lat = p_r
            p_a_lat[p_a_lat==0]=p_a_lat[p_a_lat!=0].min()
        for i,v in enumerate(small_sel):
            big_sel[i] = np.random.choice(buck_list,p=p_a_lat[buck_list,v]/\
                                     p_a_lat[buck_list,v].sum())
            buck_list = np.setdiff1d(buck_list, big_sel[i],True)
        cc_ind = np.append(big_sel[:,np.newaxis],small_sel[:,np.newaxis],axis=1)
        np.random.shuffle(cc_ind)
        Lat_M[cc_ind[:,0],cc_ind[:,1]] = 1
    
    # assign the positions of nodes
    from_ind = np.argwhere(Lat_M)
    n_of_outdeg = Lat_M.sum(axis=1)
    n_of_indeg = Lat_M.sum(axis=0)
    n_of_totdeg = n_of_indeg + n_of_outdeg
    SPD = True
    if SPD == True:
        if SPD_opt2 == 'tot':
            stand_deg = n_of_totdeg
        elif SPD_opt2 == 'out':
            stand_deg = n_of_outdeg
        elif SPD_opt2 =='in':
            stand_deg = n_of_indeg
        tmp_ind = np.random.choice(np.arange(N_cell),\
                                p=(stand_deg**2+1)/(stand_deg**2+1).sum(),\
                                    replace=False,size=N_cell)
        rand_deg = SPD_opt3
        test_distance = mnrv.pdf(All_ind)
        test_distance = test_distance/test_distance.max() + rand_deg*(np.random.rand(N_cell)-0.5)
        if SPD_opt1 == 'i':
            to_ind = test_distance.argsort()
        elif SPD_opt1 == 'p':
            to_ind = test_distance.argsort()[::-1]
    else:
        to_ind = np.arange(N_cell)
        tmp_ind = np.arange(N_cell)
        np.random.shuffle(tmp_ind)
    dest_ii = to_ind[np.argsort(tmp_ind)]
    dest_ind = dest_ii[from_ind]
    TLat_M= np.zeros((N_cell,N_cell))
    TLat_M[dest_ind[:,0],dest_ind[:,1]] = 1
    G = nx.from_numpy_matrix(TLat_M,create_using=nx.DiGraph())
    G.edges(data=True)
    for j in range(N_cell):
        G.nodes[j]['pos'] = tuple(Pos_mat[j,:])
    return csr_matrix(TLat_M),G

def avg_p_len(G_in):
    tmp_ind = []
    SPL = []
    for C in (G_in.subgraph(c).copy() for c in nx.weakly_connected_components(G_in)):
        SPL.append(nx.average_shortest_path_length(C))
        tmp_ind.append(C.number_of_nodes())
    return SPL[np.argmax(tmp_ind)]

def get_nearest_idx(vec_ref,x):
    result = []
    for i in range(vec_ref.shape[0]):
        tmp_re = np.linalg.norm(x - vec_ref[i],axis=1)
        tmp_re2 = tmp_re.argmin()
        tmp_re1 = tmp_re[tmp_re.argmin()]
        if tmp_re1 < 0.045:
            result.append(tmp_re2)
    return np.array(result).astype(int)

def f_args():
    parse = argparse.ArgumentParser(description='evaluation stable patterns')
    parse.add_argument('input_file',type=str,help='inp_coeff (npz file)')
    parse.add_argument('-d','--desination',type=str,default='/home/kh/Downloads',help='location of the output files')
    parse.add_argument('-n','--ncores',type=int,default=6,help='ncores for multiprocessing')
    return parse

if __name__ == '__main__':
    args = f_args().parse_args()
    dict_arg = vars(args)
    inp_f = dict_arg['input_file']
    tmp_Folder_str = dict_arg['desination']
    using_cores = dict_arg['ncores']

    folders = list(os.walk(tmp_Folder_str))
    for folder in folders:
        if not(folder[1] or folder[2]):
            os.rmdir(folder[0])
    
    # making directory where network data are saved
    net_folder = os.path.join(tmp_Folder_str,'Nets')
    if not(os.path.isdir(net_folder)):
        os.makedirs(net_folder)
    
    
    sel_inps = np.load(inp_f,allow_pickle=True)["sel_inps"]
    isel_inps = [sel_inps[using_cores*s:using_cores*(s+1)] for s in range(np.ceil(sel_inps.shape[0]/using_cores).astype(int))]
    tot_M = []
    clust_data = np.empty(sel_inps.shape[0])
    assort_data = np.empty((sel_inps.shape[0],4))
    len_data = np.empty(sel_inps.shape[0])
    dist_data = np.empty((sel_inps.shape[0],4))
    print(len(isel_inps))
    for iisel, isel in tqdm(enumerate(isel_inps),desc='Making_M',total=len(isel_inps)):
        with closing(Pool(using_cores)) as p:
            Tmp = p.starmap(make_net,isel)
        tmp_M = np.array([s[0] for s in Tmp])
        tmp_G = [s[1] for s in Tmp]

        # obtaining distribution parameters of distances between two connected nodes 
        indeg_dist = [np.asarray(sc_d[np.tril(inp_mat.toarray()).astype(bool)]).ravel() for inp_mat in tmp_M]
        p = Pool(using_cores)
        dist_params = p.map(beta_fit,indeg_dist)
        p.close()
        p.join()
        
        # calc clustering coefficient of generated networks
        p = Pool(using_cores)
        tmp_clust_data = np.array(p.map(get_c,tmp_G))
        p.close()
        p.join()
        
        # calc clustering coefficient of generated networks
        p = Pool(using_cores)
        tmp_assort_data = np.array(p.map(get_r,tmp_G))
        p.close()
        p.join()
        
        # calc average pathlength of generated networks
        p = Pool(using_cores)
        tmp_len_data = np.array(p.map(avg_p_len,tmp_G))
        p.close()
        p.join()
        
        clust_data[using_cores*iisel:using_cores*(iisel+1)] = tmp_clust_data
        assort_data[using_cores*iisel:using_cores*(iisel+1),:] = tmp_assort_data
        len_data[using_cores*iisel:using_cores*(iisel+1)] = tmp_len_data
        dist_data[using_cores*iisel:using_cores*(iisel+1),:] = np.c_[dist_params]
        tot_M.append(tmp_M)
        
    tot_M = [s for s in chain(*tot_M)]

    # save generated networks 
    file_name2 = os.path.join(net_folder,'network_.npz')
    np.savez(file_name2,tot_M=tot_M,sel_inps=sel_inps,\
              clust_data=clust_data, assort_data=assort_data, len_data=len_data,\
                  dist_data=dist_data)
    
