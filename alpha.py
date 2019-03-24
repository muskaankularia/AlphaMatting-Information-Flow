from math import sqrt as sqrt
import heapq
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

from lle import locally_linear_embedding as lle
from lle import barycenter_kneighbors_graph as bkg
from lle import barycenter_kneighbors_graph_ku as bkgku

from closed_form_matting import compute_weight

from sklearn.neighbors import NearestNeighbors
import numpy as np

from scipy.sparse import csr_matrix as csr
from scipy.sparse import diags
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import spsolve

from scipy.misc import imsave
import os




def cm(img, X):

	kcm = 20
	w = bkg(X, n_neighbors=kcm)
	return w
	# print(w)
	# X = np.array([[-1, -1, 2,1], [-2, -1, 4,7], [-3, -2,4,5], [1, 1,8,9], [2, 1,5,4], [3, 2,3,8]])
	# nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
	# y = nbrs._fit_X
	# print(nbrs.kneighbors(X, return_distance=False))
	# exit()
	# print(X == y)
	# pp = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(nbrs)
	# print(nbrs)
	# exit()
	# distances, indices = pp.kneighbors(X)
	# w = bkg(X, n_neighbors=2)
	# print(w)
	# print(type(w))
	# l = w[0]
	# print((l[s0]))
	# print(l[1])
	# print(l[2])
	# print(l[3])
	# print(indices)
	# print(nbrs._fit_X)


def ku(img, tmap, X):
	kku = 7

	N = X.shape[0]
	X[:,3:5] = 10*X[:,3:5]
	alpha = tmap.ravel()
	ind = np.arange(X.shape[0])

	fore = X[alpha>0.9]
	find = ind[alpha>0.9]

	back = X[alpha<0.1]
	bind = ind[alpha<0.1]

	unk = X[(alpha>0.1)&(alpha<0.9)]
	unkind = ind[(alpha>0.1)&(alpha<0.9)]

	#nearest foreground pixel to unknown
	kdt = KDTree(fore, leaf_size=30, metric='euclidean')
	nf = kdt.query(unk, k=kku, return_distance=False) 
	ind1 = find[nf]

	#nearest background pixel to unknown
	kdt = KDTree(back, leaf_size=30, metric='euclidean')
	nb = kdt.query(unk, k=kku, return_distance=False) 
	ind2 = bind[nb]

	z_ind = np.concatenate((ind1, ind2), axis = 1)
	z = X[:,:-2][z_ind]
	x_inp = unk[:,:-2]
	
	W = bkgku(x_inp, z, z_ind, n_neighbors=2*kku)
	W = W.reshape((W.shape[0], W.shape[1], 1))
	weighted_colours = W*z
	print(W.shape)
	print(z.shape)

	cpf = np.sum(weighted_colours[:,0:kku,:], axis = 1)
	cc = cpf.copy()
	cpb = np.sum(weighted_colours[:,kku:,:], axis = 1)
	cpf /= np.sum(W[:,0:kku],axis=1)
	cpb /= np.sum(W[:,kku:],axis=1)
	cpf = np.abs(cpf)
	cpb = np.abs(cpb)
	

	wf = np.zeros((N))
	wf[unkind] = np.sum(W[:,0:kku,:],axis=1)[:,0]
	wf[find] = 1
	# print(np.sum(alpha>0.1))
	# print(np.sum(wf!=0))

	H = np.sum((cpf-cpb)*(cpf-cpb),axis=1)/(3*255*255)  #2norm of cpf-cpb 
	# print(H.shape)
	# print(H[0:20])
	# q = H.toarray()
	# print(z[H>1])
	# # print(W[H>1])
	# print(W[H>1][:1])
	# print(z[H>1][:1])
	# m = W[H>1][:1] * z[H>1][:1]
	# ww = W[H>1][:1]
	# print(W[H>1][:1] * z[H>1][:1])
	# print(np.sum(m[:,0:kku,:], axis =1))
	# print(np.sum(ww[:,0:kku,:], axis =1))
	# print("---------")

	# print(cc[H>1][:1])
	# print(H[H>1][:1])
	# print(cpf[H>1][:1])
	# print(cpb[H>1][:1])
	# print(np.sum(H>1))
	H = csr((H,(unkind,unkind)),shape=(N,N))
	return wf,H
	# print(csr.count_nonzero(nu))
	# print(nu.shape)s

	# weighted_colours[0:kku,:,:] /= 

	# print(unk.shape)

	# print(ind1.shape)
	# print(ind2.shape)


	# knn = NearestNeighbors(n_neighbors + 1, n_jobs=n_jobs).fit(X)
 #    X = knn._fit_X
 #    n_samples = X.shape[0]
 #    ind = knn.kneighbors(X, return_distance=False)[:, 1:]
	
	# w = bkg(X, n_neighbors=kku)


def intra_u(img, tmap, X):

	##yet to add symmetricity
	N = X.shape[0]
	kuu = 5
	X[:,3:5] = X[:,3:5]/20
	alpha = tmap.ravel()
	
	ind = np.arange(X.shape[0])

	unk = X[(alpha>0.1)&(alpha<0.9)]
	unkind = ind[(alpha>0.1)&(alpha<0.9)]

	#nearest unknown pixels to unknown
	kdt = KDTree(unk, leaf_size=30, metric='euclidean')
	nu = kdt.query(unk, k=kuu, return_distance=False) 
	unk_nbr_true_ind = unkind[nu]
	unk_nu_ind = np.asarray([int(i/kuu) for i in range(nu.shape[0]*nu.shape[1])])
	unk_nu_true_ind = unkind[unk_nu_ind]

	nbr = unk[nu]
	nbr = np.swapaxes(nbr,1,2)
	unk = unk.reshape((unk.shape[0], unk.shape[1], 1))

	x = nbr-unk
	x = np.abs(x)
	print(x.shape)
	y = 1-np.sum(x, axis = 1)
	y[y<0] = 0
	# print(y.shape)

	row = unk_nu_true_ind
	col = unk_nbr_true_ind.ravel()
	data = y.ravel()
	z = csr((data,(col,row)),shape=(N,N))
	w = csr((data,(row,col)),shape=(N,N))
	# z = csr((data,(col,row)),shape=(h*w,h*w))
	w = w+z
	return w
	# print(csr.count_nonzero(z))
	# print(csr.count_nonzero(w))


def local(img,tmap):
	umask = (tmap>0.1) & (tmap<0.9)
	W = compute_weight(img, mask=umask, eps=10**(-7), win_rad=1).tocsr()
	return W
	# print(type(W))
	# X = csr.sum(W,axis=1)  #numpy matrix
	# D = diags(X.A.ravel()).tocsr()
	# a = csr([[1,0,0],[2,3,0],[0,0,4]])
	# b = csr([[1,0,0],[1,1,0],[0,0,1]])
	# c = a-b
	# print(c.toarray())
	# print(csr.count_nonzero(X))


def eq1(Wcm,Wuu,Wl,H,T,ak,wf):
	# sku = 0.05
	# suu = 0.01
	# sl = 1
	# lamd = 100

	sku = 0.05
	suu = 0.01
	sl = 1
	lamd = 100

	X = csr.sum(Wcm,axis=1)  #numpy matrix
	Dcm = diags(X.A.ravel()).tocsr()

	X = csr.sum(Wuu,axis=1)  #numpy matrix
	Duu = diags(X.A.ravel()).tocsr()

	X = csr.sum(Wl,axis=1)  #numpy matrix
	Dl = diags(X.A.ravel()).tocsr()

	Lifm = csr.transpose(Dcm-Wcm).dot(Dcm-Wcm) + suu*(Duu-Wuu) + sl*(Dl-Wl)
	# Lifm = suu*(Duu-Wuu) + sl*(Dl-Wl)


	A = Lifm + lamd*T + sku*H
	b = (lamd*T + sku*H).dot(wf)
	# print(csr.sum(b))
	M = diags(A.diagonal())
	# print(A.shape)
	# print(b.shape)
	alpha = cg(A, b, x0=wf, tol=1e-05, maxiter=100, M=None, callback=None, atol=None)
	# alpha = spsolve(A, b)
	# print(alpha)
	# print(type(alpha[0]))
	return alpha[0]*255
	###solve

	# A = Lifm + lamd*T
	# b = (lamd*T).dot(ak)
	###solve


def main(img_path, tri_map, save_path):

	# c = np.asarray([[1,0,0],[2,3,0],[0,0,4]])
	# d = np.asarray([[1],[1],[1]])
	# e = c.dot(d)
	# a = csr([[1,0,0],[2,3,0],[0,0,4]])
	# b = csr(e)
	# b = b.T
	# b = csr(b)
	# print(a.shape)
	# print(b.shape)
	# x = cg(a, b, x0=None, tol=1e-05, maxiter=10, callback=None, atol=None)
	# print(x)
	# exit()

	# img_path = img_path
	# tri_map = tri_map
	
	img = cv2.imread(img_path) 
	tri_map = cv2.imread(tri_map)
	tmap = tri_map[:,:,0].copy()
	tmap = tmap/255
	print(np.sum(tmap==1))

	X = []
	[h, w, c] = img.shape
	for i in range(h):
		for j in range(w):
			[r, g, b] = list(img[i, j, :])
			X.append([r, g, b, i/h, j/w])
	X = np.asarray(X)
	alpha = tmap.ravel()
	

	known = alpha.copy()
	known[(alpha>0.9)|(alpha<0.1)] = 1
	known[(alpha<0.9)&(alpha>0.1)] = 0
	T = diags(known).tocsr()
	print(T.count_nonzero())
	# print(np.sum(known==0))
	# print(X[(alpha>0.1)&(alpha<0.9)].shape[0])
	# exit()
	
	Wcm = cm(img,X)
	# Wcm = csr((h*w,h*w))
	wk,H = ku(img,tmap,X)
	# H = csr((h*w,h*w))
	# wk = csr((wk.shape))
	Wuu = intra_u(img,tmap,X)
	# Wuu = csr((h*w,h*w))
	Wl = local(img,tmap)

	ak = alpha.copy()
	ak[ak<0.9] = 0 #set all non foreground pixels to 0
	ak[ak>=0.9] = 1 #set all foreground pixels to 1

	calc_alpha = eq1(Wcm,Wuu,Wl,H,T,ak,wk)
	calc_alpha[alpha==1] = 1
	calc_alpha[alpha==0] = 0
	calc_alpha[calc_alpha>0.2] = 1
	calc_alpha[calc_alpha<0.07] = 0
	imsave(save_path, calc_alpha.reshape((h,w)))


def other():
	dira = "./data/trimap_lowres/Trimap1"
	dirb = "./data/trimap_lowres/Trimap2"
	dirc = "./data/trimap_lowres/Trimap3"
	dir1 = "./data/input_lowres"

	# dir1 = "out2/Trimap"
	l = os.listdir(dir1)

	for file in l:
		f = str(file)
		f = "troll.png"
		img_path = dir1+'/'+f
	
		save_path = "out2/Trimap1/"+f
		tri_map = dira+'/'+f
		main(img_path, tri_map, save_path)
		exit()

		save_path = "out2/Trimap2/"+f
		tri_map = dirb+'/'+f
		main(img_path, tri_map, save_path)
		
		save_path = "out2/Trimap3/"+f
		tri_map = dirc+'/'+f
		main(img_path, tri_map, save_path)


if __name__ == "__main__":
	# main()
	other()
	# cm()