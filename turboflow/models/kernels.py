import numpy as np
from numpy.core.function_base import linspace
import scipy as sp
from scipy import linalg
from scipy.linalg import solve
import matplotlib.pyplot as plt
from scipy.linalg.decomp_cholesky import cholesky

class RBFInterp(object):
    def __init__(self, eps):
        self.eps = eps
        
    def euclidean_distance(self, a, b):
        a = a[None,...].swapaxes(0,2)
        b = b.T[None,...].swapaxes(0,1)
        return np.sqrt(((a-b)**2).sum(axis=0))
        
    def fit(self, X, y):
        self.xk_ = X
        transformation = self.gauss_rbf(self.euclidean_distance(X, self.xk_))
        self.w_ = np.linalg.solve(transformation, y)
        
    def gauss_rbf(self, radius): 
        return np.exp(-1.*self.eps*(radius**2))
    
    def __call__(self, X):
        transformation = self.gauss_rbf(self.euclidean_distance(X, self.xk_))
        return transformation.dot(self.w_)


class DivFreeRBFInterp(RBFInterp):

    def __init__(self, eps):
        super(DivFreeRBFInterp, self).__init__(eps)

    def g_exp(self,x,y): return   np.exp(-self.eps*(x**2 + y**2))
    def phi11(self,x,y): return -(4.0*(self.eps**2)*(y**2)-2*self.eps) * self.g_exp(x, y)
    def phi12(self,x,y): return   4.0*(self.eps**2) * self.g_exp(x, y) * x * y
    def phi21(self,x,y): return   self.phi12(x,y)
    def phi22(self,x,y): return -(4.0*(self.eps**2)*(x**2)-2*self.eps) * self.g_exp(x, y)


    def div_free_rbf_(self, X, Xk, stage):
        N = X.shape[0]
        Nk = Xk.shape[0]
        
        # pairwise differences
        X = X[None,...].swapaxes(0,2)
        Xk = Xk.T[None,...].swapaxes(0,1)
        diff = X-Xk
        xoff = diff[0,:,:]
        yoff = diff[1,:,:]

        A11 = self.phi11(xoff, yoff) #upper left quadrant  - 11
        A12 = self.phi12(xoff, yoff) #upper right quadrant - 12
        A21 = A12                    #lower left quadrant  - 12 - equals to 21
        A22 = self.phi22(xoff,yoff)  #lower right quadrant - 22

        A = np.zeros((2*N,2*Nk))
        A[:N,:Nk] = A11
        A[N:,:Nk] = A12
        A[:N,Nk:] = A21
        A[N:,Nk:] = A22

        if stage == 'train':
            sp.linalg.cholesky(A)

        return A, (A11, A12, A22)

    def pd_inv(self, A):
        n = A.shape[0]
        I = np.identity(n)
        return sp.linalg.solve(A, I, assume_a = 'pos')

    def solve_schur_x1(self, A11, A12, A22, b1, b2):
        A22inv = self.pd_inv(A22)
        A1 = A11 - A12 @ A22inv @ A12
        b1 = b1 -  A12 @ A22inv @ b2
        return sp.linalg.solve(A1, b1, assume_a='pos')

    def fit(self, X, y):
        self.Xk_ = X
        N = X.shape[0]

        A, (A11, A12, A22) = self.div_free_rbf_(X, self.Xk_, stage='train')
        self.A_ = A
        
        d = np.zeros(2*N)
        d[:N] = y[:,0]
        d[N:] = y[:,1]

        x1 = self.solve_schur_x1(A11, A12, A22, d[:N], d[N:])
        x2 = sp.linalg.solve(A22, d[N:] - A12@x1)
        w = np.concatenate([x1, x2])
        self.w_ = w
        assert np.allclose(A.dot(w), d)

    def __call__(self, X):
        N = X.shape[0]
        B, (B11,B12,B22) = self.div_free_rbf_(X, self.Xk_, stage='test')
        ypred = B.dot(self.w_)
        return np.concatenate([ypred[:N,None], ypred[N:,None]], axis=1)


# class DivFreeRBFInterpMcNally(RBFInterp):

#     def __init__(self, eps, stencil_size=3):
#         super(DivFreeRBFInterpMcNally, self).__init__(eps)

#         nx = stencil_size
#         self.nx = nx
#         self.nd = nx*nx
#         nd = self.nd

#         A = np.zeros((2*self.nd, 2*self.nd))

#         xi = np.linspace(-(nx-1)/2, (nx-1)/2, nx)
#         self.xx, self.xy = np.meshgrid(xi, xi)
#         X = np.concatenate([self.xx.reshape(-1,1), self.xy.reshape(-1,1)], axis=1)
#         # pairwise differences
#         Xk = X.T[None,...].swapaxes(0,1)
#         X = X[None,...].swapaxes(0,2)
#         diff = X-Xk
#         xoff = diff[0,:,:]
#         yoff = diff[1,:,:]

#         A11 = self.phi11(xoff, yoff)
#         A12 = self.phi12(xoff, yoff)
#         A21 = self.phi21(xoff, yoff)
#         A22 = self.phi22(xoff, yoff)
#         A = np.block([[A11,A12],[A21,A22]])

#         self.Acho_ = sp.linalg.cho_factor(A)

#         Xk = np.zeros_like(Xk)
#         diff = X-Xk
#         xoff = diff[0,:,:]
#         yoff = diff[1,:,:]

#         T11 = np.diag(np.diag(self.phi11(xoff, yoff)))
#         T12 = np.diag(np.diag(self.phi12(xoff, yoff)))
#         T21 = np.diag(np.diag(self.phi12(xoff, yoff)))
#         T22 = np.diag(np.diag(self.phi12(xoff, yoff)))
#         self.T_ = np.block([[T11,T12],[T21,T22]])


#     def g_exp(self,x,y): 
#         r = x**2 + y**2
#         return   np.exp(-self.eps*r)

#     def phi11(self,x,y): return -(4.0*(self.eps**2)*(y**2)-2*self.eps) * self.g_exp(x, y)
#     def phi12(self,x,y): return   4.0*(self.eps**2) * self.g_exp(x, y) * x * y
#     def phi21(self,x,y): return   self.phi12(x,y)
#     def phi22(self,x,y): return -(4.0*(self.eps**2)*(x**2)-2*self.eps) * self.g_exp(x, y)

#     def __call__(self, X, y):
#         N = X.shape[0]

#         for i in range(1,N-1):

#             xy = X[i,:]
#             uv = y[i,:]



























# def __call__(self, X):
#     N = X.shape[0]
#     B, (B11,B12,B22) = self.div_free_rbf_(X, self.Xk_, stage='test')
#     ypred = B.dot(self.w_)
#     return np.concatenate([ypred[:N,None], ypred[N:,None]], axis=1)

# def __call__(self, X):
#     N = X.shape[0]
#     B = self.div_free_rbf_(X, self.Xk_)
#     ypred = B.dot(self.w_)
#     return np.concatenate([ypred[:N,None], ypred[N:,None]], axis=1)

# def div_free_rbf_matrixvalue_kernels(self, X, Xk):
#     N = X.shape[0]
#     N = Xk.shape[0]

#     rows = []
#     for i in range(N):
#         cols = []
#         for j in range(N):
#             dx = X[i,0] - Xk[j,0]
#             dy = X[i,1] - Xk[j,1]

#             a = np.zeros((2,2))
#             a[0,0] = self.phi11(dx, dy)
#             a[0,1] = self.phi12(dx, dy)
#             a[1,0] = self.phi21(dx, dy)
#             a[1,1] = self.phi22(dx, dy)

#             cols.append(a)
        
#         rows.append(cols)
    
#     A = np.block(rows)
#     sp.linalg.cholesky(A)

# def div_free_rbf(self, X, Xk):
#     N = X.shape[0]
#     A = np.zeros([2*N,2*N])

#     #upper left quadrant - 11
#     for i in range(0,N):
#         for j in range(0,N):
#             xoff = X[i,0]-Xk[j,0]
#             yoff = X[i,1]-Xk[j,1]
#             A[i,j] = self.phi11(xoff,yoff)

#     #upper right quadrant - 21
#     for i in range(0,N):
#         for j in range(N,2*N):
#             xoff = X[i,0]-Xk[j-N,0]
#             yoff = X[i,1]-Xk[j-N,1]
#             A[i,j] = self.phi21(xoff,yoff)

#     #lower left quadrant - 12
#     for i in range(N,2*N):
#         for j in range(0,N):
#             xoff = X[i-N,0]-Xk[j,0]
#             yoff = X[i-N,1]-Xk[j,1]
#             A[i,j] = self.phi12(xoff,yoff)

#     #lower right quadrant - 22
#     for i in range(N,2*N):
#         for j in range(N,2*N):
#             xoff = X[i-N,0]-Xk[j-N,0]
#             yoff = X[i-N,1]-Xk[j-N,1]
#             A[i,j] = self.phi22(xoff,yoff)

#     plt.imshow(A)
#     plt.show()

#     self.A_ = A
#     return A

# def div_free_rbf__(self, X, Xk):
#     N = X.shape[0]
    
#     Phi = np.zeros((N,N,2,2))

#     phi_prime = lambda r : -self.eps*np.exp(-self.eps*r)
#     F = lambda r : phi_prime(r)/r if r > 0 else 0
#     Fp= lambda r : (self.eps**2) * (np.exp(-self.eps*r))
#     G = lambda r : Fp(r)/r if r > 0 else 0

#     for i in range(N):
#         for j in range(N):
#             x1 = X[i,0]
#             y1 = X[i,1]
#             x2 = Xk[j,0]
#             y2 = Xk[j,1]
#             r = np.sqrt((x1-x2)**2 + (y1-y2)**2)

#             A = np.zeros((2,2))
#             A[0,0] =  (x2 - y2)**2
#             A[0,1] = -(x1 - y1)*(x2 - y2)
#             A[1,0] = -(x1 - y1)*(x2 - y2)
#             A[1,1] =  (x1 - y1)**2
#             Phi[i,j,:,:] = - F(r) * np.eye(2) - G(r) * A
    
#     A = np.zeros((2*N, 2*N))
#     A[:N,:N] = Phi[:,:,0,0]
#     A[:N,N:] = Phi[:,:,0,1]
#     A[N:,:N] = Phi[:,:,1,0]
#     A[N:,N:] = Phi[:,:,1,1]

#     plt.imshow(A)
#     plt.show()

#     return A

# def div_free_rbf_(self, X, Xk):
#     N = X.shape[0]
#     N = Xk.shape[0]

#     # xoff = X[:,0][None,...] - Xk[:,0][...,None]
#     # yoff = X[:,1][None,...] - Xk[:,1][...,None]
    
#     # pairwise differences
#     # X = X[None,...].swapaxes(0,2)
#     # Xk = Xk.T[None,...].swapaxes(0,1)
#     # diff = X-Xk
#     # xoff = diff[0,:,:]
#     # yoff = diff[1,:,:]

#     A = np.zeros((2*N,2*N))
#     A11 = np.zeros((N,N))
#     A12 = np.zeros((N,N))
#     A21 = np.zeros((N,N))
#     A22 = np.zeros((N,N))

#     for i in range(N):
#         for j in range(N):
#             dx = X[i,0] - Xk[j,0]
#             dy = X[i,1] - Xk[j,1]

#             A11[i,j] = self.phi11(dx, dy)
#             A12[i,j] = self.phi12(dx, dy)
#             A21[i,j] = A12[i,j]
#             A22[i,j] = self.phi22(dx, dy)
    
#     # A11 = self.phi11(xoff, yoff) #upper left quadrant  - 11
#     # A12 = self.phi12(xoff, yoff) #upper right quadrant - 12
#     # A21 = A12                    #lower left quadrant  - 12 - equals to 21
#     # A22 = self.phi22(xoff,yoff)  #lower right quadrant - 22

#     sp.linalg.cholesky(A11)
#     sp.linalg.cholesky(A12)
#     sp.linalg.cholesky(A22)

#     A[:N,:N] = A11
#     A[N:,:N] = A12
#     A[:N,N:] = A21
#     A[N:,N:] = A22
#     return A, (A11, A12, A22)

# def div_free_rbf_(self, X, Xk, stage):
#     N = X.shape[0]
#     Nk = Xk.shape[0]

#     # xoff = X[:,0][None,...] - Xk[:,0][...,None]
#     # yoff = X[:,1][None,...] - Xk[:,1][...,None]

#     # print(xoff.shape)
#     # print(yoff.shape)
    
#     # pairwise differences
#     X = X[None,...].swapaxes(0,2)
#     Xk = Xk.T[None,...].swapaxes(0,1)
#     diff = X-Xk
#     xoff = diff[0,:,:]
#     yoff = diff[1,:,:]

#     A = np.zeros((2*N,2*Nk))
#     A11 = np.zeros((N,Nk))
#     A12 = np.zeros((N,Nk))
#     A21 = np.zeros((N,Nk))
#     A22 = np.zeros((N,Nk))

#     # for i in range(N):
#     #     for j in range(Nk):
#     #         dx = X[i,0] - Xk[j,0]
#     #         dy = X[i,1] - Xk[j,1]

#     #         A11[i,j] = self.phi11(dx, dy)
#     #         A12[i,j] = self.phi12(dx, dy)
#     #         A21[i,j] = A12[i,j]
#     #         A22[i,j] = self.phi22(dx, dy)
    
#     A11 = self.phi11(xoff, yoff) #upper left quadrant  - 11
#     A12 = self.phi12(xoff, yoff) #upper right quadrant - 12
#     A21 = A12                    #lower left quadrant  - 12 - equals to 21
#     A22 = self.phi22(xoff,yoff)  #lower right quadrant - 22

#     A[:N,:Nk] = A11
#     A[N:,:Nk] = A12
#     A[:N,Nk:] = A21
#     A[N:,Nk:] = A22

#     # sp.linalg.cholesky(A12)
#     # sp.linalg.cholesky(A21)
#     # sp.linalg.cholesky(A11)
#     # sp.linalg.cholesky(A22)
#     if stage == 'train':
#         sp.linalg.cholesky(A)

#     return A, (A11, A12, A22)