#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 08:31:03 2020

@author: kh-lab
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import glob
from scipy import signal
from tqdm import tqdm
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, generate_binary_structure, binary_erosion, center_of_mass, sobel
from sklearn.cluster import DBSCAN, KMeans
from scipy.interpolate import griddata, interp1d
import matplotlib.patches as patches
# from matplotlib.patches import Rectangle
import hdbscan
from sklearn.preprocessing import StandardScaler
from matplotlib import colors as mcolors
from numba import jit, prange, stencil
from matplotlib.path import Path
from PIL import Image
from matplotlib import rcParams
import string
from matplotlib_scalebar.scalebar import ScaleBar
from itertools import chain
from multiprocessing import Pool
from scipy.signal import correlate, find_peaks
from scipy.spatial import ConvexHull as CH
import pandas as pd
from trackpy import link, quiet, filter_stubs, compute_drift
from trackpy.linking import Linker
from scipy.optimize import least_squares
rcParams.update({'text.usetex':True, 'font.family':'sans-serif', 'font.sans-serif':\
                  ['Helvetica'],'axes.linewidth':2})

# Change the directories of BH video and Abel's data for analysis
location_BH_vids = "/home/kh/문서/Codes/Python/SCN_modeling/Exp/"
location_Abel_data = "/home/kh/문서/Codes/Python/SCN_modeling/Exp/ABEL/Full_t/"
    
def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def detect_peaks(image):
    neighborhood = generate_binary_structure(2,4)
    local_max = maximum_filter(image, footprint=neighborhood)==image
    background = (image==0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background
    return detected_peaks

def into_Phase(array,key):
    if key == 'ab':
        kws = {'width':5,'height':0.03,'prominence':0.035}
        array = array/array.max()
    elif key == 'bh':
        kws = {'width':5}
    else:
        print('check for data set name')
    Phase_D = np.zeros(array.shape)
    for i in range(array.shape[1]):
        TTX_1_args,properties = signal.find_peaks(array[:,i],**kws)
        st_ind = np.insert(TTX_1_args+1,0,0);
        end_ind = np.insert(TTX_1_args,TTX_1_args.size,array.shape[0])
        for j in range(st_ind.size):
            Phase_D[st_ind[j]:end_ind[j],i] = np.linspace(0,2*np.pi,end_ind[j]-st_ind[j])
    return Phase_D



def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(1).mean(-1)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def cart2pol_pos_a(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    phi[phi < 0] += np.pi
    return(rho, phi)


# import BH's videos 
cap = []
for i in range(3):
    cap.append(cv2.VideoCapture(location_BH_vids + 'exp_' + str(i+2) +'.mov'))
    

n_of_frames = np.zeros(3).astype(int)
H = np.zeros(3).astype(int)
W = np.zeros(3).astype(int)
for i in range(3):
    n_of_frames[i] = int(cap[i].get(cv2.CAP_PROP_FRAME_COUNT))
    H[i] = cap[i].get(cv2.CAP_PROP_FRAME_HEIGHT)
    W[i] = cap[i].get(cv2.CAP_PROP_FRAME_WIDTH)

frame = [np.zeros((int(a),int(b),int(c))).astype(int) for a,b,c in zip(H,W,n_of_frames)]
for i in range(3):
    flag = 0
    while True:
        _,tmp_frame1 = cap[i].read()
        if tmp_frame1 is None:
            break
        if i == 0:
            tmp_frame1[0:35:,160:285,:]=0
        elif i == 1:
            tmp_frame1[0:35:,160:285,:]=0
            tmp_frame1[35:67:,195:280,:]=0
        elif i == 2:
            tmp_frame1[0:35:,160:285,:]=0
        frame[i][:,:,flag] = tmp_frame1[:,:,2]
        flag += 1
    cap[i].release()

data = [np.zeros((int(a),int(b),int(c))) for a,b,c in zip(H/10,W/10,n_of_frames)]
for i in range(3):
    for j in range(n_of_frames[i]):
        data[i][:,:,j] = rebin(frame[i][:,:,j],[int(H[i]/10),int(W[i]/10)])
t_data = [np.arange(0,s/6/24,step=1/6/24) for s in n_of_frames]
ex_c_ind = [np.argwhere(s.sum(axis=2) > 15e+4) for s in data]
ex_non_c_ind = [np.argwhere(s.sum(axis=2) <= 15e+4) for s in data]

shock_idx = np.array([348,480])
shock_t = t_data[0][shock_idx]

snd_shock_idx = np.array([887,1380])
snd_shock_t = t_data[0][snd_shock_idx]

tmp_data = []
for i in range(3):
    data[i][ex_non_c_ind[i][:,0],ex_non_c_ind[i][:,1],:] = np.nan
    tmp_data.append(data[i][ex_c_ind[i][:,0],ex_c_ind[i][:,1],:]/255)



# filtering data & exporting synchrony
filt_b, filt_a = signal.butter(2,1/50)
filt_b2, filt_a2 = signal.butter(2,1/100)
filt_D = []
Z_filt_D = []
BH_img = []
Tot_filt_sig = []
Phase_D_BH = []
synchrony_BH = []
filtered_synchrony_BH = []
std_phase_BH = []
filtered_std_phase_BH = []
for i in range(3):
    inp_sig = tmp_data[i]
    filt_sig = signal.filtfilt(filt_b,filt_a,inp_sig,axis=1,padlen=30)
    phase_data = into_Phase(filt_sig.T,'bh')
    phase_sig = ((np.cos(phase_data)+1)/2).T
    Tot_filt_sig.append(filt_sig)
    Ztmp_filt_D = np.zeros(data[i].shape)
    Btmp_filt_D = np.zeros(data[i].shape)
    Btmp_filt_D[:] = np.nan
    Ztmp_filt_D[ex_c_ind[i][:,0],ex_c_ind[i][:,1],:] = phase_sig
    Btmp_filt_D[ex_c_ind[i][:,0],ex_c_ind[i][:,1],:] = phase_sig
    synchrony_BH.append(np.abs(np.sum(np.exp(1j*phase_data),axis=1))/phase_data.shape[1])
    filtered_synchrony_BH.append(signal.filtfilt(filt_b2,filt_a2,synchrony_BH[i],method='gust'))
    std_phase_BH.append(phase_data.std(axis=1))
    filtered_std_phase_BH.append(signal.filtfilt(filt_b2,filt_a2,std_phase_BH[i],method='gust'))
    Phase_D_BH.append(phase_sig)
    Z_filt_D.append(Ztmp_filt_D)
    BH_img.append(Btmp_filt_D)
    tmp_filt_D = np.zeros(data[i].shape)
    tmp_filt_D[:] = np.nan
    tmp_filt_D[ex_c_ind[i][:,0],ex_c_ind[i][:,1],:] = (filt_sig-filt_sig.min())/(filt_sig-filt_sig.min()).max()*255
    filt_D.append(tmp_filt_D)


# area calculation from scalebar
DV_len = np.array([418,490,562])
max_cI = np.array([np.logical_not(np.isnan(s[:,:,1])).sum(axis=0).max() for s in filt_D])
box_len = DV_len/max_cI
n_of_cell = np.array([s.shape[0] for s in ex_c_ind])

val_Area = box_len**2*n_of_cell
sc_len = np.sqrt(val_Area/n_of_cell)
s_c_ind = [s*sc_len[i] for i,s in enumerate(ex_c_ind)]


def get_single_RFpv(DP_arr):
    tmp_im = correlate(DP_arr,DP_arr,'full')
    tmp_xcor_img  = tmp_im/tmp_im.max()
    peak_pos = np.argwhere(detect_peaks(tmp_xcor_img))
    mov_peak_pos = peak_pos-np.array(DP_arr.shape)+1
    tmp_r,tmp_th = cart2pol_pos_a(mov_peak_pos[:,1],mov_peak_pos[:,0])
    tmp_polar = np.append(tmp_r[:,np.newaxis],tmp_th.T[:,np.newaxis],axis=1)
    tmp_peak_data = np.append(tmp_polar,tmp_xcor_img[peak_pos[:,0],peak_pos[:,1]][:,np.newaxis],axis=1)
    peak_pos[:,[0, 1]] = peak_pos[:,[1, 0]]
    cond_1_ind = tmp_peak_data[:,2] > 0.25
    cond_2_ind = np.round(tmp_peak_data[:,0],1) != 0
    cond_ind = np.logical_and(cond_1_ind,cond_2_ind)
    tmp_RFpv = tmp_peak_data[cond_ind,:][:,[0,2]]
    tmp_x = tmp_RFpv[tmp_RFpv[:,1].argsort()[::-1],0]
    tmp_y = tmp_RFpv[tmp_RFpv[:,1].argsort()[::-1],1]
    if tmp_x.size !=0:
        RFpv = np.append(tmp_x[0,np.newaxis],tmp_y[0,np.newaxis],axis=0)
    else:
        RFpv = np.array([[],[]])
    return RFpv

@jit('float64(float64,float64)')
def D_phi(a,b):
    a = 2*np.pi*(a-0.5)
    b = 2*np.pi*(b-0.5)
    if np.abs(a-b) <= np.pi:
        return a-b
    elif a-b > 0:
        return a-b-2*np.pi
    else:
        return a-b+2*np.pi
    
@stencil
def kernel1(a):
    result = D_phi(a[-1,0],a[-1,-1])+\
        D_phi(a[-1,1],a[-1,0])+\
            D_phi(a[0,1],a[-1,1])+\
                D_phi(a[1,1],a[0,1])+\
                    D_phi(a[1,0],a[1,1])+\
                        D_phi(a[1,-1],a[1,0])+\
                            D_phi(a[0,-1],a[1,-1])+\
                                D_phi(a[-1,-1],a[0,-1])
    return result
@stencil
def kernel2(a):
    result = D_phi(a[-2,-1],a[-2,-2])+\
        D_phi(a[-2,0],a[-2,-1])+\
            D_phi(a[-2,1],a[-2,0])+\
            D_phi(a[-2,2],a[-2,1])+\
            D_phi(a[-1,2],a[-2,2])+\
            D_phi(a[0,2],a[-1,2])+\
            D_phi(a[1,2],a[0,2])+\
            D_phi(a[2,2],a[1,2])+\
            D_phi(a[2,1],a[2,2])+\
            D_phi(a[2,0],a[2,1])+\
            D_phi(a[2,-1],a[2,0])+\
            D_phi(a[2,-2],a[2,-1])+\
            D_phi(a[1,-2],a[2,-2])+\
            D_phi(a[0,-2],a[1,-2])+\
            D_phi(a[-1,-2],a[0,-2])+\
            D_phi(a[-2,-2],a[-1,-2])
    return result


@jit(nopython=True,parallel=True)
def get_phase_v(a,chkern=1):
    V1 = np.empty(a.shape)
    if chkern == 1:
        for j in prange(a.shape[-1]):
                V1[:,:,j] = kernel1(a[:,:,j])
    elif chkern == 2:
        for j in prange(a.shape[-1]):
                V1[:,:,j] = kernel2(a[:,:,j])
    return V1

def cont_path(im):
    tmp_im = np.array(im*255, dtype = np.uint8)
    contours,_= cv2.findContours(tmp_im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    test_arr = np.array(contours,dtype=object)
    cont_1 = np.append(np.squeeze(test_arr[1]),test_arr[1][0,0].reshape(1,2),axis=0)
    path = Path(cont_1)
    return path

def cont_bounds(im):
    tmp_im = np.array(im*255, dtype = np.uint8)
    contours,_=cv2.findContours(tmp_im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    test_arr = np.squeeze(np.array(contours,dtype=object))
    path = Path(np.insert(test_arr,0,test_arr[-1],axis=0))
    return path 


def ex_points(path, test_points):
    result_b = np.empty(test_points.shape[0],dtype=bool)
    for i in range(test_points.shape[0]):
        result_b[i] = path.contains_point(test_points[i])
    return result_b

def get_bg(arr):
    return np.any(arr[:,:,:10]!=0,axis=2)

def get_PS(inp_V,bgd_img):
    Sbgd_img = binary_erosion(bgd_img)
    s_x = sobel(Sbgd_img,axis=0,mode='constant')
    s_y = sobel(Sbgd_img,axis=1,mode='constant')
    edge_img = np.hypot(s_x,s_y) > 0.3
    tmp_path = cont_path(edge_img)
    PS_coord = np.array([[0,0,0,0]])
    tmp_pV = np.isclose(inp_V,2*np.pi,rtol=1e-2).astype(np.int8)
    tmp_mV = np.isclose(inp_V,-2*np.pi,rtol=1e-2).astype(np.int8)
    for i in range(tmp_pV.shape[-1]):
        tmp_limg1,tmp_n1 = label(tmp_pV[:,:,i])
        tmp_limg2,tmp_n2 = label(tmp_mV[:,:,i])
        c_l1 = np.arange(tmp_n1+1)[np.unique(tmp_limg1,return_counts=True)[1]>2]
        c_l2 = np.arange(tmp_n2+1)[np.unique(tmp_limg2,return_counts=True)[1]>2]
        tmp_coo1 = np.asarray(center_of_mass(tmp_pV[:,:,i],tmp_limg1,index=c_l1))
        tmp_coo2 = np.asarray(center_of_mass(tmp_mV[:,:,i],tmp_limg2,index=c_l2))
        if tmp_coo1.size!=0:
            coo1 = tmp_coo1[ex_points(tmp_path, tmp_coo1[:,::-1])]
            if coo1.size !=0:
                coo1 = np.concatenate((coo1,np.ones((coo1.shape[0],1))*i,np.ones((coo1.shape[0],1))),axis=1)
                PS_coord = np.append(PS_coord,coo1,axis=0)
        if tmp_coo2.size !=0:
            coo2 = tmp_coo2[ex_points(tmp_path, tmp_coo2[:,::-1])]
            if coo2.size !=0:
                coo2 = np.concatenate((coo2,np.ones((coo2.shape[0],1))*i,np.zeros((coo2.shape[0],1))),axis=1)
                PS_coord = np.append(PS_coord,coo2,axis=0)
    PS_coord = np.delete(PS_coord,0,axis=0)
    return PS_coord

def m_getps(using_core,V_imgs,B_imgs):
    from multiprocessing import Pool
    p = Pool(using_core)
    tmp_PS_c = p.starmap(get_PS,zip(V_imgs,B_imgs))
    p.close()
    p.join()
    return np.array(tmp_PS_c,dtype=object)

# searching the locations of phase singularities
tmp_filtered_peak_v = []
for j in range(3):
    tmp_imgs = np.moveaxis(Z_filt_D[j],-1,0)
    p = Pool(12)
    tmp_filtered_peak_v.append(p.map(get_single_RFpv,tmp_imgs))
    p.close()
    p.join()

# removing meaningless phase singularities
filtered_peak_v = []
for i in range(3):
    tmp_coo = tmp_filtered_peak_v[i]
    tmp_data = np.empty((1,3))
    for j,tc_i in enumerate(tmp_coo):
        if tc_i.size !=0:
            tmp_data = np.append(tmp_data, np.insert(tc_i,0,j*t_data[0][1])[np.newaxis,:],axis=0)
    tmp_data = np.delete(tmp_data,0,axis=0)
    filtered_peak_v.append(tmp_data)
clustering = [DBSCAN(eps=1.5,min_samples=10).fit(filtered_peak_v[s]) for s in range(3)]

def plot_ps(X,cdata=None):
    fig = plt.figure()
    ax1 = fig.add_subplot(111,projection='3d')
    if cdata is None:
        ax1.scatter(X[:,0],X[:,1],X[:,2],s=1)
    else:
        ax1.scatter(X[:,0],X[:,1],X[:,2],c=cdata,s=1)
    return fig, ax1

cnorm = plt.Normalize()
def movie_saver(arr,video_filename,coord = None):
    rs = 3
    height = arr.shape[0]*rs
    width = arr.shape[1]*rs
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    fps = 30
    
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    if coord is None:
        for i in tqdm(range(arr.shape[-1])):
            gray = cv2.normalize(arr[:,:,i], None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            gray = cv2.resize(gray,None,fx=rs,fy=rs,interpolation=cv2.INTER_NEAREST)
            im_color = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
            out.write(im_color)
    else:
        color_m = (cm.tab20(cnorm(np.arange(0,coord[:,3].max()+1)))[:,:3]*255).astype(np.uint8)
        np.random.shuffle(color_m)
        for i in tqdm(range(arr.shape[-1])):
            gray = cv2.normalize(arr[:,:,i], None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            im_color = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
            m_coord = coord[coord[:,2] == i,:2][:,::-1]
            c_code = coord[coord[:,2] == i,3].astype(int)
            for j,point in enumerate(m_coord):
                im_color = cv2.circle(im_color, tuple(point.astype(int)),1,color_m[c_code[j]].tolist(),thickness=-1)
            im_color = cv2.resize(im_color,None,fx=rs,fy=rs,interpolation=cv2.INTER_NEAREST)
            out.write(im_color)
    out.release()

def Cvt_Arr2Img(arr,md='g'):
    arr_size = arr.shape
    if md == 'c':
        tmp_img = []
        for i in range(arr_size[-1]):
            ttmp_img = Image.fromarray((np.stack((arr[:,:,i],)*3,axis=-1)*255).astype(np.uint8),'HSV')
            tmp_img.append(np.array(ttmp_img))
    elif md== 'g':
        tmp_img = []
        for i in range(arr_size[-1]):
            ttmp_img = Image.fromarray((arr[:,:,i]*255).astype(np.uint8),'L')
            tmp_img.append(np.array(ttmp_img))
    return np.array(tmp_img)

def calc_vecF(arr,jump_f=2):
    Z_img = arr[:,:,::jump_f]
    bg_img = get_bg(arr)
    st_r,st_c = np.argwhere(bg_img).min(axis=0) 
    en_r,en_c = np.argwhere(bg_img).max(axis=0) + np.ones(2,dtype=int)
    told_img = Z_img[st_r:en_r,st_c:en_c,:-1]
    tnew_img = Z_img[st_r:en_r,st_c:en_c,1:]
    DA = np.zeros(Z_img[:,:,0].shape).astype(bool)
    DA[st_r:en_r,st_c:en_c] = True
    coord_life = np.argwhere(bg_img)
    coord_tot = np.argwhere(DA)
    tmp_bix = np.array([s.tolist() in coord_life.tolist() for s in coord_tot])
    XX = coord_tot[:,0]
    YY = coord_tot[:,1]
    old_img = Cvt_Arr2Img(told_img)
    new_img = Cvt_Arr2Img(tnew_img)
    u = [];
    v = [];
    mean_v = np.empty((Z_img.shape[-1]-1,2))
    for i in range(Z_img.shape[-1]-1):
        tmp_uv = cv2.calcOpticalFlowFarneback(old_img[i], new_img[i], None, pyr_scale = 0.5, levels = 3, winsize = 7, iterations = 10, poly_n = 7, poly_sigma = 1.5, flags = 0)
        tmp_u = tmp_uv[:,:,0].ravel()
        tmp_v = tmp_uv[:,:,1].ravel()
        tmp_u[~tmp_bix] = 0
        tmp_v[~tmp_bix] = 0
        u.append(tmp_u)
        v.append(-tmp_v)
        mean_v[i] = np.array([tmp_u.mean(), -tmp_v.mean()])
    tmp_ind = np.linalg.norm(mean_v,axis=1) > 0.02
    Big_v = mean_v.copy()
    Big_v[~tmp_ind] = 0
    tmp_MB_v = Big_v[:,:].mean(axis=0)
    MB_v = cart2pol(tmp_MB_v[0], tmp_MB_v[1])
    time = np.arange(0,(arr.shape[-1]-1)*1/6/24,1/6/24*jump_f)
    return YY,XX,u,v,MB_v, mean_v, time

def show_vecF(imgarr,arr,jump_f=2):
    tmp_x,tmp_y = (np.argwhere(get_bg(arr)).min(axis=0) + np.argwhere(get_bg(arr)).max(axis=0))/2
    tmp_d = calc_vecF(arr,jump_f)
    XX = tmp_d[0]
    YY = tmp_d[1]
    u = tmp_d[2]
    v = tmp_d[3]
    fig_D = plt.figure()
    ax = fig_D.add_subplot(1,1,1)
    im = ax.imshow(imgarr[:,:,0],animated=True,alpha=0.7,cmap='hsv')
    im.set_clim([0,1])
    im2 = ax.quiver(XX,YY,u[0],v[0],scale=9)
    ax.axis('off')
    def animate2(i):
        im.set_data(imgarr[:,:,15::2][:,:,i])
        im2.set_UVC(u[15+i],v[15+i])
    anim_D = animation.FuncAnimation(fig_D, animate2, interval=30, frames=235,
                                      repeat_delay = 50)
    return anim_D, fig_D



# import Abel's data
files = glob.glob(location_Abel_data+'*.csv')
files.sort()
fulldata_files = files[::2]
locdata_files = files[1::2]

fulldata = []
locdata = []
N_cells = np.zeros(5).astype(int)
T_len = np.zeros(5).astype(int)
for i in range(5):
    fulldata.append(np.loadtxt(fulldata_files[i],dtype=np.float32,delimiter=','))
    locdata.append(np.loadtxt(locdata_files[i],dtype=np.float32,delimiter=','))
    N_cells[i] = locdata[i].shape[0]
    T_len[i] = fulldata[i].shape[0]
tmp_TTX_idx = np.array([90,109,82,107,113])
tmp_Wash_idx = np.array([234,252,224,240,242])

def rot_mat(th):
    theta = th/360*2*np.pi
    return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

# split data by region
rot_th = np.array([0,180,-90,-45,90+151.85])#degree
split_x = np.array([362,-425,-308,-8,-402])
coord = []
for i in range(5):
    coord.append(locdata[i].dot(rot_mat(rot_th[i])))

Left_c = []
Right_c = []
for i in range(5):
    Left_c.append(np.argwhere(coord[i][:,0] > split_x[i]))
    Right_c.append(np.argwhere(coord[i][:,0] <= split_x[i]))

tmp_t_ab_data = []
loc = []
tmp_ab_data = []
for i in range(5):
    tmp_loc_1 = np.c_[coord[i][Left_c[i],0],coord[i][Left_c[i],1]]
    tmp_loc_1 = tmp_loc_1 - np.mean(tmp_loc_1,axis=0)
    tmp_loc_2 = np.c_[coord[i][Right_c[i],0],coord[i][Right_c[i],1]]
    tmp_loc_2 = tmp_loc_2 - np.mean(tmp_loc_2,axis=0)
    loc.append(tmp_loc_1)
    loc.append(tmp_loc_2)
    tmp_data_1 = fulldata[i][:,Left_c[i].ravel()]
    tmp_data_2 = fulldata[i][:,Right_c[i].ravel()]
    tmp_ab_data.append(tmp_data_1)
    tmp_ab_data.append(tmp_data_2)
    tmp_t_ab_data.append(np.linspace(0,tmp_data_1.shape[0]/24,tmp_data_1.shape[0]))
    tmp_t_ab_data.append(np.linspace(0,tmp_data_1.shape[0]/24,tmp_data_1.shape[0]))

n_of_cell_ab = [s.shape[0] for s in loc]
clustering_locs = [DBSCAN(eps=50,min_samples=5).fit(loc[s]) for s in range(10)]
tmp_ab_data = [tmp_ab_data[s][:,clustering_locs[s].labels_==0] for s in range(10)]
loc = [loc[s][clustering_locs[s].labels_==0] for s in range(10)]


def interP(tarr,arr):
    newt = np.arange(tarr[0],tarr[-1],step=1/6/24)
    newarr = interp1d(tarr,arr,kind='cubic',axis=0)(newt)
    return newt,newarr

def find_nearest2(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# interpolate Abel's data
TTX_t = tmp_t_ab_data[0][tmp_TTX_idx]
Wash_t = tmp_t_ab_data[0][tmp_Wash_idx]
ab_data = []
t_ab_data = []
for i in range(10):
    tmp_t,tmp_d = interP(tmp_t_ab_data[i], tmp_ab_data[i])
    ab_data.append(tmp_d)
    t_ab_data.append(tmp_t)

TTX_time = [find_nearest2(t_ab_data[0],TTX_t[s]) for s in range(5)]
Wash_time = [find_nearest2(t_ab_data[0],Wash_t[s]) for s in range(5)]

    
def into_Amp(array):
    tmp_amp = np.empty((array.shape))
    tmp_t = np.linspace(0,array.shape[0]/24,array.shape[0])
    for i in range(array.shape[1]):
        peak_ind,_ = signal.find_peaks(array[:,i],width=5)
        tmp_amp[:,i] = np.interp(tmp_t,tmp_t[peak_ind],array[peak_ind,i])
    return tmp_amp


# get the amplitude, phase, synchrony from time series data
phase_data = []
amp_data = []
synchrony_ab_origin = []
for i in range(10):
    phase_data.append(into_Phase(ab_data[i],'ab'))
    amp_data.append(into_Amp(ab_data[i]))
    synchrony_ab_origin.append(np.abs(np.sum(np.exp(1j*phase_data[i]),axis=1))/phase_data[i].shape[1])

cos_p = [np.cos(s) for s in phase_data]
tmp_y = [np.moveaxis(s,0,1)[:,:TTX_time[i//2]] for i,s in enumerate(cos_p)]
aa = [interp1d(t_ab_data[s][:TTX_time[s//2]],tmp_y[s],'cubic',axis=1) for s in range(10)]
tmp_xx = [np.arange(0,t_ab_data[s][:TTX_time[s//2]][-1],0.01) for s in range(10)]
tmp_yy = [aa[s](tmp_xx[s])for s in range(10)]
loc_ab = [np.array(s) for s in loc]
t_max = [signal.find_peaks(tmp_yy[s].std(axis=0))[0][3] for s in range(10)]
dab = [tmp_yy[s][:,t_max[s]+15] for s in range(10)]
X = [np.append(loc_ab[s],dab[s][:,np.newaxis],axis=1) for s in range(10)]

# get_ core shell label
stdsc = StandardScaler()
kmc = KMeans(n_clusters=2)

C_label = []
S_label = []
for s in range(10):
    scaled_X = stdsc.fit_transform(X[s])
    kmc.fit(scaled_X)
    labels = np.argwhere(kmc.labels_)
    flabels = np.argwhere(np.logical_not(kmc.labels_))
    if loc_ab[s][labels].mean(axis=0).ravel()[1] > loc_ab[s][flabels].mean(axis=0).ravel()[1]:
        S_label.append(labels)
        C_label.append(flabels)
    else:
        S_label.append(flabels)
        C_label.append(labels)
        
def calc_sarea(c_arr,s_arr,t_arr):
    tmp_sig = s_arr - c_arr
    tmp_ind = np.argwhere(tmp_sig[:-1]*tmp_sig[1:] < 0)
    low_ind = tmp_ind[c_arr[tmp_ind] < 0]
    high_ind = np.setdiff1d(tmp_ind, low_ind)
    high_ind = np.delete(high_ind,np.argwhere(high_ind < low_ind[0]))
    size_ind = min(high_ind.size,low_ind.size)
    tmp_A = 0
    for ind_i in range(size_ind-1):
        tmp_A += np.trapz(tmp_sig[low_ind[ind_i]:high_ind[ind_i]],dx=t_arr[1]*24)
    sig_darea = tmp_A/(size_ind-1)
    return sig_darea

tmp_sarea = np.empty(10)
for i in range(10):
    c_arr = tmp_yy[i][C_label[i].ravel(),:].mean(axis=0)
    s_arr = tmp_yy[i][S_label[i].ravel(),:].mean(axis=0)
    t_arr = tmp_xx[i]
    tmp_sarea[i] = calc_sarea(c_arr, s_arr, t_arr)

tot_loc = loc[0]
for i in range(9):
    tot_loc = np.r_[tot_loc,loc[i+1]]

rep_ind = np.repeat(np.arange(0,5),2)
mean_amps = np.empty((10,3))
ratio_amp = np.zeros(10)
for i,j in enumerate(rep_ind):
    mean_amps[i,0] = amp_data[i][:TTX_time[j],:].mean()
    mean_amps[i,1] = amp_data[i][TTX_time[j]:Wash_time[j],:].mean()
    mean_amps[i,2] = amp_data[i][Wash_time[j]:,:].mean()
    ratio_amp[i] = mean_amps[i,1]/mean_amps[i,0]

def PolyArea2D(pts):
    lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
    area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
    return area

ab_area = np.zeros(10)
for i in range(10):
    ab_area[i] = PolyArea2D(loc[i][CH(loc[i]).vertices])

def calc_period_e(t_arr,arr):
    arr = ttx_pd - np.pi
    t_arr = t_pd
    fft = np.fft.rfft(arr,axis=1)
    abs_fft = np.abs(fft)
    [_,N] = fft.shape
    f = np.linspace(0, 1/(t_arr[1]-t_arr[0]), N)
    tmp_y = abs_fft[:,:N // 2]/N
    max_ind = np.argmax(tmp_y,axis=1)
    tmp_f = f[:N // 2];
    Periods = 1/tmp_f[max_ind]*2
    return Periods

#  get periods from time series data
j = 0
ttx_pd = phase_data[j][:,TTX_time[j]:Wash_time[j]]
t_pd = t_ab_data[j][TTX_time[j]:Wash_time[j]]*24
period_ttx = []
period_pre = []
period_pst = []
for j in range(10):
    ttx_pd = phase_data[j][:,TTX_time[j//2]:Wash_time[j//2]]
    t_pd = t_ab_data[j][TTX_time[j//2]:Wash_time[j//2]]*24
    pre_pd = phase_data[j][:,:TTX_time[j//2]]
    aft_pd = phase_data[j][:,Wash_time[j//2]:]
    TTX_Period = np.array([])
    for i in range(ttx_pd.shape[1]):
        TTX_Period = np.append(TTX_Period,np.diff(np.argwhere(ttx_pd[:,i]==np.pi*2).ravel()).astype(int))
    Pre_Period = np.array([])
    for i in range(pre_pd.shape[1]):
        Pre_Period = np.append(Pre_Period,np.diff(np.argwhere(pre_pd[:,i]==np.pi*2).ravel()).astype(int))
    PST_Period = np.array([])
    for i in range(aft_pd.shape[1]):
        PST_Period = np.append(PST_Period,np.diff(np.argwhere(aft_pd[:,i]==np.pi*2).ravel()).astype(int))
    period_ttx.append(TTX_Period)
    period_pre.append(Pre_Period)
    period_pst.append(PST_Period)


tot_period_ttx = np.array([])
for s in range(10):
    tot_period_ttx = np.append(tot_period_ttx,period_ttx[s])


def get_grid_D(loc,phi,time):
    x = loc[:,0]
    y = -loc[:,1]
    z = (np.cos(phi)+1)/2
    z_ori = phi
    x_fine = np.arange(min(x)-50, max(x)+50, step=10)
    y_fine = np.arange(min(y)-50, max(y)+50, step=10)
    x_grid, y_grid = np.meshgrid(x_fine, y_fine)
    z_grid = np.empty(x_grid.shape + tuple([time.size]))
    z_grid_ori = np.empty(x_grid.shape + tuple([time.size]))
    for i in range(time.size):
        z_grid[:,:,i] = griddata((x, y), z[i,:], (x_grid.ravel(), y_grid.ravel()), method='linear').reshape(x_grid.shape)
        z_grid_ori[:,:,i] = griddata((x, y), z_ori[i,:], (x_grid.ravel(), y_grid.ravel()), method='linear').reshape(x_grid.shape)
        tmp_grid = z_grid_ori[:,:,i].copy()
        tmp_phase_D = tmp_grid[np.logical_not(np.isnan(z_grid_ori[:,:,i]))]
        if i==0:
            phase_D = tmp_phase_D[np.newaxis,:]
        else:
            phase_D = np.append(phase_D,tmp_phase_D[np.newaxis,:],axis=0)
    xcorr_ab = z_grid.copy()
    xcorr_ab[np.isnan(z_grid)] = 0            
    return xcorr_ab, z_grid, phase_D, z_grid_ori

# convert Abel's data to 2d images for correlation length of Abel's data
p = Pool(10)
out = p.starmap(get_grid_D,zip(loc,phase_data,t_ab_data))
p.close()
p.join()
for_xcorr_ab = [s[0] for s in out]
im_for_anim = [s[1] for s in out]
T_phase_D = [s[2] for s in out]
im_for_anim2 = [s[3] for s in out]


# two types of synchrony (grid and point version)
grid_synchrony = []
filt_b,filt_a = signal.butter(2,1/50)
filtered_grid_sync = []
filtered_sync_ab = []
for i in range(10):
    grid_synchrony.append(np.abs(np.sum(np.exp(1j*T_phase_D[i]),axis=1))/T_phase_D[i].shape[1])
    filtered_grid_sync.append(signal.filtfilt(filt_b,filt_a,grid_synchrony[i],method='gust'))
    filtered_sync_ab.append(signal.filtfilt(filt_b,filt_a,synchrony_ab_origin[i],method='gust'))

# calculate phase singularities
std_phase_ab = []
filtered_std_phase_ab = []
for i in range(10):
    std_phase_ab.append(T_phase_D[i].std(axis=1))
    filtered_std_phase_ab.append(signal.filtfilt(filt_b,filt_a,std_phase_ab[i],method='gust'))

tmp_filtered_peak_v_ab = []
for j in tqdm(range(10)):
    tmp_imgs = np.moveaxis(for_xcorr_ab[j],-1,0)
    p = Pool(12)
    tmp_filtered_peak_v_ab.append(p.map(get_single_RFpv,tmp_imgs))
    p.close()
    p.join()

filtered_peak_v_ab = []
for i in range(10):
    tmp_coo = tmp_filtered_peak_v_ab[i]
    tmp_data = np.empty((1,3))
    for j,tc_i in enumerate(tmp_coo):
        if tc_i.size !=0:
            tmp_data = np.append(tmp_data, np.insert(tc_i,0,j*t_ab_data[0][1])[np.newaxis,:],axis=0)
    tmp_data = np.delete(tmp_data,0,axis=0)
    filtered_peak_v_ab.append(tmp_data)


sc_len_ab = 10
t_TTX = np.repeat(TTX_time,2)
t_Wash = np.repeat(Wash_time,2)
ttx_all_peaks = np.array([])
before_all_peaks = np.array([])
ttx_sep_peaks = []
N_cell_ab = np.array([s.shape[0] for s in loc])
bool_idx = np.argwhere(N_cell_ab > 100).ravel()
for i in bool_idx:
    con_ind_1 = filtered_peak_v_ab[i][:,0]/(1/6/24) > t_TTX[i]
    con_ind_2 = filtered_peak_v_ab[i][:,0]/(1/6/24) < t_Wash[i]
    con_ind = np.logical_and(con_ind_1,con_ind_2)
    ttx_all_peaks = np.append(ttx_all_peaks,filtered_peak_v_ab[i][con_ind,1]*sc_len_ab)
    ttx_sep_peaks.append(filtered_peak_v_ab[i][con_ind,1]*sc_len_ab)
corr_len_ab  = np.array([np.mean(s) for s in ttx_sep_peaks])
np.mean(corr_len_ab/np.sqrt(ab_area))
bgd_bh = [get_bg(s) for s in Z_filt_D]
bgd_ab = [get_bg(s) for s in for_xcorr_ab]

ref_V = []
for tmp_D in Z_filt_D:
    ref_V.append(get_phase_v(tmp_D,1))
tmp_PS = m_getps(3,ref_V,bgd_bh)
tmp_PS[1] = tmp_PS[1][tmp_PS[1][:,-2] >= shock_idx[1]]

ab_ref_V = []
for tmp_D in for_xcorr_ab:
    ab_ref_V.append(get_phase_v(tmp_D))
ab_PS = m_getps(10,ab_ref_V,bgd_ab)


# calculate optical flow of Abel's 2-d image sequence
tmp_vec_inf1 = []
tmp_vec_inf1.append(calc_vecF(Z_filt_D[0]))
tmp_vec_inf1.append(calc_vecF(Z_filt_D[1]))
p = Pool(10)
tmp_vec_inf2 = p.map(calc_vecF,for_xcorr_ab)
p.close()
p.join()
vec_inf = tmp_vec_inf1 + tmp_vec_inf2



def link_D(test_data,filt_num=100):
    tmp_z = test_data[:,3].astype(bool)
    x1 = test_data[tmp_z,0].astype(int)
    y1 = test_data[tmp_z,1].astype(int)
    x2 = test_data[~tmp_z,0].astype(int)
    y2 = test_data[~tmp_z,1].astype(int)
    t1 = test_data[tmp_z,2]
    t2 = test_data[~tmp_z,2]
    df_a = pd.DataFrame({'y':y1,'x':x1,'frame':t1})
    df_b = pd.DataFrame({'y':y2,'x':x2,'frame':t2})
    Linker.MAX_SUB_NET_SIZE = 100
    quiet()
    tmp_link_a = link(df_a,search_range=10,memory=100,\
                    link_strategy='numba',neighbor_strategy='KDTree')
    tmp_link_b = link(df_b,search_range=10,memory=100,\
                    link_strategy='numba',neighbor_strategy='KDTree')
    tmp_link_b['particle'] += tmp_link_a['particle'].max()+1
    link_data_a = filter_stubs(tmp_link_a,filt_num)
    link_data_b = filter_stubs(tmp_link_b,filt_num)
    link_data = pd.concat([link_data_a,link_data_b],ignore_index=True)

    lk_group = link_data.groupby('particle')
    hold_time = np.empty(lk_group.ngroups)
    mov_distance = np.empty(lk_group.ngroups)
    shifted_d = np.empty(lk_group.ngroups)
    particle_label = np.empty(lk_group.ngroups)
    start_time = np.empty(lk_group.ngroups)
    count_p = np.empty(lk_group.ngroups)
    for ii,(i,tmp_g) in enumerate(lk_group):
        mov_distance[ii] = np.linalg.norm(compute_drift(tmp_g).to_numpy()[:,:2],axis=1).sum()
        shifted_d[ii] = np.linalg.norm(compute_drift(tmp_g).to_numpy()[:,:2].sum(axis=0))
        hold_time[ii] = tmp_g['frame'].max()-tmp_g['frame'].min()
        start_time[ii] = tmp_g['frame'].min()
        particle_label[ii] = i
        count_p[ii] = tmp_g.shape[0]
    particle_frame = pd.DataFrame({'particle':particle_label,'start_time':start_time,'hold_time':hold_time,'shifted_d':shifted_d,'count_p':count_p})
    new_X = link_data[['x','y','frame']].to_numpy()
    new_X = np.append(new_X,link_data['particle'].factorize()[0][:,np.newaxis],axis=1)
    return new_X, link_data_a['particle'].max(), particle_frame, link_data

# link all phase singularities
PS_data1,D_max1,pinfo_df1,link_a = link_D(tmp_PS[0],5)
PS_data2,D_max2,pinfo_df2,link_b = link_D(tmp_PS[1],5)


tmp_dd = [pinfo_df1,pinfo_df2]
test_cp = [s['count_p'].to_list()  for s in tmp_dd if s.size != 0]
test_ht = [s['hold_time'].to_list()  for s in tmp_dd if s.size != 0]
test_cp = np.array(list(chain(*test_cp)))
test_ht = np.array(list(chain(*test_ht)))

# %% Correlation length Exp plot(only bh)
from matplotlib import gridspec as gs
from matplotlib import rcParams
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
rcParams.update({'text.usetex':True, 'font.family':'sans-serif', 'font.sans-serif':\
                  ['Helvetica'],'axes.linewidth':2,'axes.labelsize':20})

j = 2
fig = plt.figure(figsize=(19,5))
Grids = gs.GridSpec(nrows=1,ncols=5,width_ratios=[1,1,1,1.5,1.5])
ax = []
for i in Grids:
    ax.append(fig.add_subplot(i))
ii = 1230
ax[0].imshow(BH_img[j][:,:,35],cmap='hsv')
ax[1].imshow(BH_img[j][:,:,ii],cmap='hsv')
ax[1].add_artist(ScaleBar(sc_len[2],'um'))
tmp_img = Z_filt_D[j][:,:,ii]
corr_img = correlate(tmp_img,tmp_img)[21:42+22,14:28+15]
corr_gimg = (corr_img/corr_img.max()*255).astype(np.uint8)
img2 = clahe.apply(corr_gimg)
corr_inferno_img = cv2.applyColorMap(img2, cv2.COLORMAP_INFERNO)
corr_inferno_img = cv2.cvtColor(corr_inferno_img, cv2.COLOR_BGR2RGB)

peak_pos = np.argwhere(detect_peaks(corr_img))
tmp_peak_data = corr_img[peak_pos[:,0],peak_pos[:,1]]
peak_pos = np.delete(peak_pos,tmp_peak_data.argmax(),axis=0)
tmp_peak_data = np.delete(tmp_peak_data,tmp_peak_data.argmax(),axis=0)
ax[2].imshow(corr_inferno_img)
sc = ax[2].scatter(peak_pos[:,1],peak_pos[:,0],c=tmp_peak_data,cmap='magma')
sc.set_clim([38,38.5])


ax[-1].set_box_aspect(1)
for i in range(3):
    ax[i].axis('off')


x = filtered_peak_v[j][:,0]
y = filtered_peak_v[j][:,1]*box_len[j]


scaler = StandardScaler()
data = scaler.fit_transform(filtered_peak_v[j])
clusterer = hdbscan.HDBSCAN(min_cluster_size=10).fit(data)
cond_idx = clusterer.outlier_scores_ < np.quantile(clusterer.outlier_scores_,0.7)
x = x[cond_idx]
y = y[cond_idx]
z = clusterer.outlier_scores_[cond_idx]

nbins = 10
n, _ = np.histogram(x, bins=nbins)
sy, _ = np.histogram(x, bins=nbins, weights=y)
sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
mean = sy / n
std = np.sqrt(sy2/n - mean*mean)


ax[3].scatter(x, y)
ax[3].errorbar((_[1:] + _[:-1])/2, mean, yerr=std,linewidth=2,capsize=5,color='C1')
ax[3].set_xlabel(r'\textbf{Time (Days)}',fontsize=20)
ax[3].set_ylabel(r'\textbf{Correlation length (}${\mu}\textrm{m}$\textbf{)}',fontsize=20)
ax[3].set_box_aspect(1)
ax[3].tick_params(length=5,width=2,labelsize=18)
ax[3].set_yticks(np.arange(150,500,100))


test_st = np.array([])
test_ttx = np.array([])
for i in range(10):
    test_st = np.append(test_st,synchrony_ab_origin[i][50:TTX_time[i//2]-72].mean())
    test_ttx = np.append(test_ttx,synchrony_ab_origin[i][TTX_time[i//2]+50:Wash_time[i//2]-50].mean())

labels=[r'\textbf{Before TTX}',r'\textbf{After TTX}']

ax[4].set_ylabel(r'$r_\textrm{sync}$',fontsize=20)
boxprops = dict(linestyle='-', linewidth=2)
bplot1 = ax[4].boxplot([test_st,test_ttx],
                         vert=True,  
                         patch_artist=True,  
                         labels=labels,
                         showfliers=False,
                         widths=(0.5),
                         boxprops=boxprops)  

ax[4].tick_params(length=5,width=2,labelsize=18)
ax[4].set_box_aspect(1)


for i,ax_i in enumerate(ax[1:]):
    ax_i.text(-0.1, 1.05, r'\textbf{'+string.ascii_lowercase[i]+'}', transform=ax_i.transAxes, 
        size=30, weight='bold')
fig.tight_layout()


# %% Direction of wave patterns (Abel data)
fig = plt.figure(figsize=[18.48/2,9.77])
ax = []
for i in range(12):
    ax.append(fig.add_subplot(3,4,i+1))
    ax[i].axis('off')
fig.subplots_adjust(wspace=0.001,hspace=0.001)
tmp_idx = np.array([130,180,220,220,270,270,\
                    125,125,250,250,210,210])
draw_imgset = []
draw_imgset.append(BH_img[0])
draw_imgset.append(BH_img[1])
for i in range(10):
    draw_imgset.append(im_for_anim[i])
    
for i in range(12):
    ax[i].imshow(draw_imgset[i][:,:,tmp_idx[i]],cmap='hsv')
    ax[i].set_xlim([0,40])
    ax[i].set_ylim([40,0])
    ax[i].set_aspect('equal')


from matplotlib_scalebar.scalebar import ScaleBar
st_t = np.append(shock_t,TTX_t.repeat(2))
sclen_list = np.append(sc_len[:2],np.ones(10)*10)
dominant_direction=np.empty((12,2))
for i in range(12):
    XX = vec_inf[i][0]
    YY = vec_inf[i][1]
    u = vec_inf[i][2]
    v = vec_inf[i][3]
    BV = vec_inf[i][4]
    mean_v = vec_inf[i][5]
    tmp_t = vec_inf[i][-1]
    bool_idx = tmp_t < st_t[i]
    stVec = mean_v[bool_idx[:mean_v.shape[0]]].mean(axis=0)
    stVec = stVec/np.linalg.norm(stVec)
    dominant_direction[i] = stVec
    qv = ax[i].quiver(XX.mean(),YY.mean(),stVec[0],stVec[1],width=0.11,lw=2,scale=4 ,fc='w',ec='k')
    qv.set_clim([-np.pi,np.pi])
    sb = ScaleBar(sclen_list[i],'um')
    ax[i].add_artist(sb)

fig.tight_layout()
    

# %% fitting ps

test_a1 = np.zeros(t_data[1].size)
for i in pinfo_df1.index:
    tmp_d = np.zeros(test_a1.shape)
    tmp_idx = np.arange(pinfo_df1['start_time'][i],(pinfo_df1['hold_time']+pinfo_df1['start_time'])[i],dtype=int)
    tmp_d[tmp_idx]  = 1
    test_a1 += tmp_d

test_a2 = np.zeros(t_data[1].size)
for i in pinfo_df2.index:
    tmp_d = np.zeros(test_a2.shape)
    tmp_idx = np.arange(pinfo_df2['start_time'][i],(pinfo_df2['hold_time']+pinfo_df2['start_time'])[i],dtype=int)
    tmp_d[tmp_idx]  = 1
    test_a2 += tmp_d
    
    


peak_max = np.zeros(2,dtype=int)
peak_ii, prop_ii = find_peaks(test_a1,width=2,prominence=2)
max_idx = prop_ii['prominences'].argmax()
peak_max[0] = peak_ii[max_idx]
peak_ii, prop_ii = find_peaks(test_a2,width=2,prominence=2)
max_idx = prop_ii['prominences'].argmax()
peak_max[1] = peak_ii[max_idx]

def dec_exp(x,t,y):
    return x[0]*np.exp(-t/x[1]) + x[2] - y

def get_tau(arr,peak_idx):
    tmp_tt = np.arange(0,2000,step=1/6/24)[:arr.size]
    tmp_t = (tmp_tt[peak_idx:]-tmp_tt[peak_idx])
    tmp_y = arr[peak_idx:]
    lsqp = least_squares(dec_exp, x0=[tmp_y[0],10,0.1], args=(tmp_t,tmp_y),\
                     bounds=([tmp_y[0]-1,0.1,0],[tmp_y[0]+1,100,1]))
    return lsqp.x, lsqp.success

parlq1,_ = get_tau(test_a1,peak_max[0])
parlq2,_ = get_tau(test_a2,peak_max[1])

# %% number of PS (BH's data)
sample_idx = 1
fig = plt.figure(figsize=(19,5))
Grids = gs.GridSpec(nrows=1,ncols=4,width_ratios=[1,1,1,1.5])
ax = []
for i in Grids:
    ax.append(fig.add_subplot(i))
for i in range(3):
    ax[i].set_aspect('equal')
ax = np.array(ax)
ax[0].axis('off')
ax[1].axis('off')
ax[1].add_artist(ScaleBar(sc_len[sample_idx],'um',fixed_value=100))

ax[2].axis('off')
link_gr = link_b.groupby('particle')
long_particle = pinfo_df2.particle[pinfo_df2['hold_time'] > 0]
gdf=[]
for i in long_particle:
    gdf.append(link_gr.get_group(i))

shock_ii = shock_idx[1]
shock_ii_s = shock_ii-80
shock_ii_e = shock_ii+25
ax[0].imshow(BH_img[sample_idx][:,:,shock_ii_s],cmap='hsv')
ax[1].imshow(BH_img[sample_idx][:,:,shock_ii_e],cmap='hsv')
edge_p = patches.PathPatch(cont_bounds(bgd_bh[sample_idx]), facecolor='none',lw=3)
ax[2].add_patch(edge_p)
ax[2].invert_yaxis()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
k = 0
for i in range(long_particle.size):
    x = gdf[i].x.to_numpy()
    y = gdf[i].y.to_numpy()
    t = gdf[i].frame
    z = (gdf[i].frame.to_numpy() - gdf[i].frame.min())/(gdf[i].frame.to_numpy() - gdf[i].frame.min()).max()
    tmp_flag = (shock_ii_e >= gdf[i].frame.to_numpy().min()) & ((shock_ii_e <= gdf[i].frame.to_numpy().max()))
    fil_idx = find_nearest(t,shock_ii_e)
    if tmp_flag:
        sc1 = ax[1].scatter(y[fil_idx],x[fil_idx],s=150,ec='k',lw=2,zorder=10)
        sc2 = ax[2].scatter(y[fil_idx],x[fil_idx],s=150,ec='k',lw=2,zorder=10,alpha=0.7)
        for ii in range(x.size-1):
            ax[2].plot(y[ii:ii+2],x[ii:ii+2],alpha=z[ii],lw=3,color=sc2.get_facecolor()[0][:3])
        k += 1


ax[3].plot(t_data[1],test_a2,label='sample2',lw=2)
ax[3].plot(t_data[1][peak_max[1]:],dec_exp(parlq2,t_data[1][:t_data[1][peak_max[1]:].size],\
                                         np.zeros(t_data[1][peak_max[1]:].size)),lw=3)
ax[3].set_ylabel(r'\textbf{Number of singularities}', fontsize=20,fontweight='bold')
ax[3].set_xlabel(r'\textbf{Time (Days)}', fontsize=20,fontweight='bold')
ax[3].tick_params(labelsize=15,width=2,length=5)
rect_2_1 = plt.Rectangle([shock_t[1],-1],6/24,30,facecolor='r',alpha=0.5)
rect_2_2 = plt.Rectangle([snd_shock_t[1],-1],6/24,30,facecolor='r',alpha=0.5)
ax[3].add_patch(rect_2_1)
ax[3].add_patch(rect_2_2)
ax[3].vlines(t_data[1][peak_max[1]],-1,20,color='k',linestyle='--')

ax[3].set_xlim([0,13])
ax[3].set_ylim([-0.2,21.5])
ax[3].set_box_aspect(1)
ax[3].scatter(t_data[1][shock_ii_s],test_a2[shock_ii_s],marker='*',s=300,c='C3',zorder=3)
ax[3].scatter(t_data[1][shock_ii_e],test_a2[shock_ii_e],marker='*',s=300,c='C3',zorder=3)


for i,ax_i in enumerate(ax[1:]):
    ax_i.text(-0.1, 1.05, r'\textbf{'+string.ascii_lowercase[i]+'}', transform=ax_i.transAxes, 
        size=30, weight='bold')
fig.tight_layout()



# %% PS animation (single)
g_link = link_b.groupby('particle')
p1 = []
intp = []
ps_xyt = []
for i in g_link.count().index:
    tmp_df = g_link.get_group(i)
    p1.append(tmp_df)
    data_xy = np.c_[tmp_df.y.to_numpy(),tmp_df.x.to_numpy()]
    data_t = tmp_df.frame.to_numpy()
    tmp_intp = interp1d(data_t,data_xy.T)
    tmp_t = np.arange(data_t.min(),data_t.max()+1)
    tmp_y = tmp_intp(tmp_t).T
    ps_xyt.append(np.c_[data_t,data_xy])
p1 = np.array(p1,dtype=object)



fig_A = plt.figure(1,figsize=(3.56,5.43))
fig_A.patch.set_alpha(0.)
ax = plt.axes()
r_ind = 1
im = ax.imshow(np.random.rand(int(H[r_ind]/10),int(W[r_ind]/10)),animated=True,cmap='gray')
sc = []
tmp_clist = np.array([s for s in mcolors.CSS4_COLORS.keys()])
tmp_clist = tmp_clist[np.random.randint(0,tmp_clist.size,p1.shape[0])]
for s in range(p1.shape[0]):
    sc.append(ax.scatter(5,5,s=150,c=tmp_clist[s],ec='k',linewidth=2))

ax.axis('off')
ax.set_xlim([0,W[r_ind]/10])
ax.set_ylim([H[r_ind]/10,0])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
time_text = ax.text(.5, 1.5, '0', fontsize=15,bbox=props)
    
def animate(i):
    ii = 3*i
    im.set_data(BH_img[r_ind][:,:,ii])
    for j in range(p1.shape[0]):
        if ii in ps_xyt[j][:,0]:
            jj = np.argwhere(ps_xyt[j][:,0]==ii)[0][0]
            sc[j].set_offsets(ps_xyt[j][jj,1:])
        else:
            sc[j].set_offsets([-5,-5])
    if ii >= shock_idx[r_ind] and ii < snd_shock_idx[r_ind]:
        time_text.set_color('orange')
    if ii >= snd_shock_idx[r_ind]:
        time_text.set_color('red')
    time_text.set_text(f'{t_data[r_ind][ii]:.2f} Days')
    return im
anim_A = animation.FuncAnimation(fig_A, animate, interval=20, frames=int(n_of_frames[r_ind]/3),
                                 repeat_delay = 500)
# anim_A.save('/home/kh/문서/Data/scn_data/ps.gif',fps=20)