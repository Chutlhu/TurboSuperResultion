#A class for RBF-FD divergence-free derivatives
# Colin P. McNally 2010, 2011, 2019
# colin@colinmcnally.ca
#
#  Queen Mary, University of London
#
#  Formerly
#
#  Department of Astrophysics
#  American Museum of Natural History
#  79th Street at Central Park West
#  New York, NY, USA, 10024
#
#  and
#
#  Department of Astronomy
#  Columbia University
#
# This code is not very fast, its here to show you how things can be done.
#
# GPL-3.0-or-later
#

import numpy as np
from scipy import linalg

class rbf_vector_divfree:
    #defualt RBF bandwidth
    eps = (1.0/8.0)**2 

    ##### Scalar radial basis function#####
    def phi(self,x,y):
        eps = self.eps 
        return np.exp(-eps*(x**2+y**2))

    def phix(self,x,y):
        eps = self.eps 
        #these comment lines are Maxima computer algebra system code
        #f90(factor(diff(exp(-eps*(x**2+y**2)),x)));
        return -2*eps*x*np.exp(-eps*y**2-eps*x**2)

    def phiy(self,x,y):
        eps = self.eps 
        #f90(factor(diff(exp(-eps*(x**2+y**2)),y)));
        return -2*eps*y*np.exp(-eps*y**2-eps*x**2)

    def phixx(self,x,y):
        eps = self.eps 
        #f90(factor(diff(diff(exp(-eps*(x**2+y**2)),x),x)));
        return 2*eps*(2*eps*x**2-1)*np.exp(-eps*y**2-eps*x**2)

    def phixy(self,x,y):
        eps = self.eps 
        #f90(factor(diff(diff(exp(-eps*(x**2+y**2)),x),y)));
        return 4*eps**2*x*y*np.exp(-eps*y**2-eps*x**2)

    def phiyy(self,x,y):
        eps = self.eps 
        #f90(factor(diff(diff(exp(-eps*(x**2+y**2)),y),y)));
        return 2*eps*(2*eps*y**2-1)*np.exp(-eps*y**2-eps*x**2)



    #####Divergence free matrix valued basis function and derivatives#####
    def phi22(self,x,y):
        eps = self.eps 
        return -(4*eps**2*x**2-2*eps)*np.exp(-eps*(x**2+y**2))

    def phi22x(self,x,y):
        eps = self.eps 
        return 4*eps**2*x*(2*eps*x**2-3)*np.exp(-eps*y**2-eps*x**2)

    def phi22y(self,x,y):
        eps = self.eps 
        return 4*eps**2*(2*eps*x**2-1)*y*np.exp(-eps*y**2-eps*x**2)

    def phi22xx(self,x,y):
        eps = self.eps 
        #f90(factor(diff(diff( -(4*eps**2*x**2-2*eps)*exp(-eps*(x**2+y**2)),x),x)));
        return -4*eps**2*(4*eps**2*x**4-12*eps*x**2+3)*np.exp(-eps*y**2-eps*x**2)

    def phi22xy(self,x,y):
        eps = self.eps 
        #f90(factor(diff(diff( -(4*eps**2*x**2-2*eps)*exp(-eps*(x**2+y**2)),x),y)));
        return -8*eps**3*x*(2*eps*x**2-3)*y*np.exp(-eps*y**2-eps*x**2)

    def phi22yy(self,x,y):
        eps = self.eps 
        #f90(factor(diff(diff( -(4*eps**2*x**2-2*eps)*exp(-eps*(x**2+y**2)),y),y)));
        return -4*eps**2*(2*eps*x**2-1)*(2*eps*y**2-1)*np.exp(-eps*y**2-eps*x**2)

    def phi12(self,x,y):
        eps = self.eps 
        return 4.0*eps**2*np.exp(-eps*(x**2+y**2))*x*y

    def phi12x(self,x,y):
        eps = self.eps 
        return -4*eps**2*(2*eps*x**2-1)*y*np.exp(-eps*y**2-eps*x**2)

    def phi12y(self,x,y):
        eps = self.eps 
        return -4*eps**2*x*(2*eps*y**2-1)*np.exp(-eps*y**2-eps*x**2)

    def phi12xx(self,x,y):
        eps = self.eps 
        #f90(factor(diff(diff(4.0*eps**2*exp(-eps*(x**2+y**2))*x*y,x),x)));
        return 8*eps**3*x*(2*eps*x**2-3)*y*np.exp(-eps*y**2-eps*x**2)

    def phi12xy(self,x,y):
        eps = self.eps 
        #f90(factor(diff(diff(4.0*eps**2*exp(-eps*(x**2+y**2))*x*y,x),y)));
        return 4*eps**2*(2*eps*x**2-1)*(2*eps*y**2-1)*np.exp(-eps*y**2-eps*x**2)

    def phi12yy(self,x,y):
        eps = self.eps 
        #f90(factor(diff(diff(4.0*eps**2*exp(-eps*(x**2+y**2))*x*y,y),y)));
        return 8*eps**3*x*y*(2*eps*y**2-3)*np.exp(-eps*y**2-eps*x**2)

    def phi11(self,x,y): 
        eps = self.eps 
        return -(4*eps**2*y**2-2*eps)*np.exp(-eps*(x**2+y**2))

    def phi11x(self,x,y):
        eps = self.eps 
        return 4*eps**2*x*(2*eps*y**2-1)*np.exp(-eps*y**2-eps*x**2)

    def phi11y(self,x,y):
        eps = self.eps 
        return 4*eps**2*y*(2*eps*y**2-3)*np.exp(-eps*y**2-eps*x**2)

    def phi11xx(self,x,y): 
        eps = self.eps 
        #f90(factor(diff(diff(-(4*eps**2*y**2-2*eps)*exp(-eps*(x**2+y**2)),x),x)));
        return -4*eps**2*(2*eps*x**2-1)*(2*eps*y**2-1)*np.exp(-eps*y**2-eps*x**2)

    def phi11xy(self,x,y): 
        eps = self.eps 
        #f90(factor(diff(diff(-(4*eps**2*y**2-2*eps)*exp(-eps*(x**2+y**2)),x),y)));
        return -8*eps**3*x*y*(2*eps*y**2-3)*np.exp(-eps*y**2-eps*x**2)

    def phi11yy(self,x,y): 
        eps = self.eps 
        #f90(factor(diff(diff(-(4*eps**2*y**2-2*eps)*exp(-eps*(x**2+y**2)),y),y)));
        return -4*eps**2*(4*eps**2*y**4-12*eps*y**2+3)*np.exp(-eps*y**2-eps*x**2)

    def phi21(self,x,y):
        return self.phi12(x,y)

    def phi21x(self,x,y):
        return self.phi12x(x,y)

    def phi21y(self,x,y):
        return self.phi12y(x,y)

    def phi21xx(self,x,y):
        return self.phi12xx(x,y)

    def phi21xy(self,x,y):
        return self.phi12xy(x,y)

    def phi21yy(self,x,y):
        return self.phi12yy(x,y)

    ### End basis functions

    def __init__(self,nx,eps_override = None):
        #precompute the interpolation matrix for the set of points we want to use
        if (eps_override):
            self.eps = eps_override

        #this first block sets up A, the div-free interpolation matrix, 4 quadrants for 4 values in the matrix valued RBF
        self.nx = nx
        self.nd = nx*nx
        nd = self.nd
        A = np.zeros([2*self.nd,2*self.nd])
        ti = np.linspace(-(nx-1)/2, (nx-1)/2, nx)
        self.x, self.y = np.meshgrid(ti, ti)
        x = self.x
        y = self.y
        #upper left quadrant - 11
        for i in range(0,nd):
            for j in range(0,nd):
                xoff = x.flatten()[i]-x.flatten()[j]
                yoff = y.flatten()[i]-y.flatten()[j]
                A[i,j] = self.phi11(xoff,yoff)

        #upper right quadrant - 21
        for i in range(0,nd):
            for j in range(nd,2*nd):
                xoff = x.flatten()[i]-x.flatten()[j-nd]
                yoff = y.flatten()[i]-y.flatten()[j-nd]
                A[i,j] = self.phi21(xoff,yoff)

        #lower left quadrant - 12
        for i in range(nd,2*nd):
            for j in range(0,nd):
                xoff = x.flatten()[i-nd]-x.flatten()[j]
                yoff = y.flatten()[i-nd]-y.flatten()[j]
                A[i,j] = self.phi12(xoff,yoff)

        #lower right quadrant - 22
        for i in range(nd,2*nd):
            for j in range(nd,2*nd):
                xoff = x.flatten()[i-nd]-x.flatten()[j-nd]
                yoff = y.flatten()[i-nd]-y.flatten()[j-nd]
                A[i,j] = self.phi22(xoff,yoff)

        self.Acho = linalg.cho_factor(A)
        #self.Alu = linalg.lu_factor(A)

        #initialize for scalar derivatives - note the matrix is smaller here
        As = np.zeros([nd,nd])
        for i in range(0,nd):
            for j in range(0,nd):
                xoff = x.flatten()[i]-x.flatten()[j]
                yoff = y.flatten()[i]-y.flatten()[j]
                As[i,j] = self.phi(xoff,yoff)

        self.Ascho = linalg.cho_factor(As)
        #self.Aslu = linalg.lu_factor(As)

        #tabulte coefficients for stencil
        self.tabphi11 = np.zeros([self.nd,self.nd])
        self.tabphi21 = np.zeros([self.nd,self.nd])
        self.tabphi12 = np.zeros([self.nd,self.nd])
        self.tabphi22 = np.zeros([self.nd,self.nd])
        self.tabphi11x = np.zeros([self.nd,self.nd])
        self.tabphi21x = np.zeros([self.nd,self.nd])
        self.tabphi12x = np.zeros([self.nd,self.nd])
        self.tabphi22x = np.zeros([self.nd,self.nd])
        self.tabphi11y = np.zeros([self.nd,self.nd])
        self.tabphi21y = np.zeros([self.nd,self.nd])
        self.tabphi12y = np.zeros([self.nd,self.nd])
        self.tabphi22y = np.zeros([self.nd,self.nd])
        self.tabphi11xx = np.zeros([self.nd,self.nd])
        self.tabphi21xx = np.zeros([self.nd,self.nd])
        self.tabphi12xx = np.zeros([self.nd,self.nd])
        self.tabphi22xx = np.zeros([self.nd,self.nd])
        self.tabphi11xy = np.zeros([self.nd,self.nd])
        self.tabphi21xy = np.zeros([self.nd,self.nd])
        self.tabphi12xy = np.zeros([self.nd,self.nd])
        self.tabphi22xy = np.zeros([self.nd,self.nd])
        self.tabphi11yy = np.zeros([self.nd,self.nd])
        self.tabphi21yy = np.zeros([self.nd,self.nd])
        self.tabphi12yy = np.zeros([self.nd,self.nd])
        self.tabphi22yy = np.zeros([self.nd,self.nd])
        
        xn = 0.0
        yn = 0.0
        xoff = np.zeros([self.nd])
        yoff = np.zeros([self.nd])

        for i in range(0,self.nd):
            xoff[i] = -(self.x.flatten()[i]-xn)
            yoff[i] = -(self.y.flatten()[i]-yn)

        for i in range(0,self.nd):
            #just the value
            self.tabphi11[i,i] = self.phi11(xoff[i],yoff[i]) 
            self.tabphi21[i,i] = self.phi21(xoff[i],yoff[i]) 
            self.tabphi12[i,i] = self.phi12(xoff[i],yoff[i])
            self.tabphi22[i,i] = self.phi22(xoff[i],yoff[i])

            #x derivative
            self.tabphi11x[i,i] = self.phi11x(xoff[i],yoff[i])
            self.tabphi21x[i,i] = self.phi21x(xoff[i],yoff[i])
            self.tabphi12x[i,i] = self.phi12x(xoff[i],yoff[i])
            self.tabphi22x[i,i] = self.phi22x(xoff[i],yoff[i])

            #y derivative
            self.tabphi11y[i,i] = self.phi11y(xoff[i],yoff[i])
            self.tabphi21y[i,i] = self.phi21y(xoff[i],yoff[i])
            self.tabphi12y[i,i] = self.phi12y(xoff[i],yoff[i])
            self.tabphi22y[i,i] = self.phi22y(xoff[i],yoff[i])

            #xx derivative
            self.tabphi11xx[i,i] = self.phi11xx(xoff[i],yoff[i])
            self.tabphi21xx[i,i] = self.phi21xx(xoff[i],yoff[i])
            self.tabphi12xx[i,i] = self.phi12xx(xoff[i],yoff[i])
            self.tabphi22xx[i,i] = self.phi22xx(xoff[i],yoff[i])

            #xy derivative
            self.tabphi11xy[i,i] = self.phi11xy(xoff[i],yoff[i])
            self.tabphi21xy[i,i] = self.phi21xy(xoff[i],yoff[i])
            self.tabphi12xy[i,i] = self.phi12xy(xoff[i],yoff[i])
            self.tabphi22xy[i,i] = self.phi22xy(xoff[i],yoff[i])

            #yy derivative
            self.tabphi11yy[i,i] = self.phi11yy(xoff[i],yoff[i])
            self.tabphi21yy[i,i] = self.phi21yy(xoff[i],yoff[i])
            self.tabphi12yy[i,i] = self.phi12yy(xoff[i],yoff[i])
            self.tabphi22yy[i,i] = self.phi22yy(xoff[i],yoff[i])


        self.tabphi = np.zeros([self.nd,self.nd])
        self.tabphix = np.zeros([self.nd,self.nd])
        self.tabphiy = np.zeros([self.nd,self.nd])
        self.tabphixx = np.zeros([self.nd,self.nd])
        self.tabphixy = np.zeros([self.nd,self.nd])
        self.tabphiyy = np.zeros([self.nd,self.nd])
        
        for i in range(0,self.nd):
            self.tabphi[i,i] = self.phi(xoff[i],yoff[i])
            self.tabphix[i,i] = self.phix(xoff[i],yoff[i])
            self.tabphiy[i,i] = self.phiy(xoff[i],yoff[i])
            self.tabphixx[i,i] = self.phixx(xoff[i],yoff[i])
            self.tabphixy[i,i] = self.phixy(xoff[i],yoff[i])
            self.tabphiyy[i,i] = self.phiyy(xoff[i],yoff[i])

        #end initialization

    def scalarderivs(self,v,dx):
        #take a scalar derivative
        d = np.zeros([self.nd])
        #v0 = v.mean()
        v0 = v[(self.nx-1)//2,(self.nx-1)//2]
        v = v - v0

        #solve the interpolation problem for the data values given
        d[0:v.size] = v.flatten()
        #self.cs = linalg.lu_solve(self.Aslu,d)
        self.cs = linalg.cho_solve(self.Ascho,d)
        phivals  = np.zeros([self.nd])
        phixvals  = np.zeros([self.nd])
        phiyvals  = np.zeros([self.nd])
        phixxvals  = np.zeros([self.nd])
        phixyvals  = np.zeros([self.nd])
        phiyyvals  = np.zeros([self.nd])

        #set up to sum the vale and derivatives of each of the basis functions at (0,0) 
        #   the place where we are taking the derivative
        # see Bayona et al. section 2 for a very similar thing
        for i in range(0,self.nd):
            phivals[i] = self.tabphi[i,i] * self.cs[i]
            phixvals[i] = self.tabphix[i,i] * self.cs[i]
            phiyvals[i] = self.tabphiy[i,i] * self.cs[i]
            phixxvals[i] = self.tabphixx[i,i] * self.cs[i]
            phixyvals[i] = self.tabphixy[i,i] * self.cs[i]
            phiyyvals[i] = self.tabphiyy[i,i] * self.cs[i]

        comp = sum(phivals)
        compx = sum(phixvals)
        compy = sum(phiyvals)
        compxx = sum(phixxvals)
        compxy = sum(phixyvals)
        compyy = sum(phiyyvals)

        #add back the mean
        comp = comp + v0

        #scale for a grid of spacing dx
        fdf = 1.0/dx
        sdf = 1.0/(dx*dx)
    
        #these are the derivatives
        compx = compx*fdf
        compy = compy*fdf

        compxx = compxx*sdf
        compxy = compxy*sdf
        compyy = compyy*sdf

        return [compx,compy,compxx,compxy,compyy]


    def divfreederivs(self,bx,by,dx):
        #calculate the derivatives of a div-free vector field
        d = np.zeros([2*self.nd])
        #bx0 = bx.mean()
        #by0 = by.mean()
        bx0 = bx[(self.nx-1)//2,(self.nx-1)//2] # central point of the stencil
        by0 = by[(self.nx-1)//2,(self.nx-1)//2] # central point of the stencil
        bx = bx - bx0
        by = by - by0
        #here I'm setting up the rhs of the interpolation problem, note I put both the x and y components of the data in
        d[0:bx.size] = bx.flatten()
        d[bx.size:]  = by.flatten()
        #solve the interpolation problem
        #self.c = linalg.lu_solve(self.Alu,d)
        self.c = linalg.cho_solve(self.Acho,d)

        #now we've calculated the coefficients to interpolate the given data, do sums
        # over the basis functions and the derivatives of the basis functions to get
        # the derivatives of the interpolated vector field
        phivals  = np.zeros([self.nd,4])
        phixvals  = np.zeros([self.nd,4])
        phiyvals  = np.zeros([self.nd,4])
        phixxvals  = np.zeros([self.nd,4])
        phixyvals  = np.zeros([self.nd,4])
        phiyyvals  = np.zeros([self.nd,4])
      
        for i in range(0,self.nd):
        #just the value
            phivals[i,0] = self.tabphi11[i,i] * self.c[i]
            phivals[i,1] = self.tabphi21[i,i] * self.c[i+self.nd]
            phivals[i,2] = self.tabphi12[i,i] * self.c[i]
            phivals[i,3] = self.tabphi22[i,i] * self.c[i+self.nd]

            #x derivative
            phixvals[i,0] = self.tabphi11x[i,i] * self.c[i]
            phixvals[i,1] = self.tabphi21x[i,i] * self.c[i+self.nd]
            phixvals[i,2] = self.tabphi12x[i,i] * self.c[i]
            phixvals[i,3] = self.tabphi22x[i,i] * self.c[i+self.nd]

            #y derivative
            phiyvals[i,0] = self.tabphi11y[i,i] * self.c[i]
            phiyvals[i,1] = self.tabphi21y[i,i] * self.c[i+self.nd]
            phiyvals[i,2] = self.tabphi12y[i,i] * self.c[i]
            phiyvals[i,3] = self.tabphi22y[i,i] * self.c[i+self.nd]

            #xx derivative
            phixxvals[i,0] = self.tabphi11xx[i,i] * self.c[i]
            phixxvals[i,1] = self.tabphi21xx[i,i] * self.c[i+self.nd]
            phixxvals[i,2] = self.tabphi12xx[i,i] * self.c[i]
            phixxvals[i,3] = self.tabphi22xx[i,i] * self.c[i+self.nd]

            #xy derivative
            phixyvals[i,0] = self.tabphi11xy[i,i] * self.c[i]
            phixyvals[i,1] = self.tabphi21xy[i,i] * self.c[i+self.nd]
            phixyvals[i,2] = self.tabphi12xy[i,i] * self.c[i]
            phixyvals[i,3] = self.tabphi22xy[i,i] * self.c[i+self.nd]

            #yy derivative
            phiyyvals[i,0] = self.tabphi11yy[i,i] * self.c[i]
            phiyyvals[i,1] = self.tabphi21yy[i,i] * self.c[i+self.nd]
            phiyyvals[i,2] = self.tabphi12yy[i,i] * self.c[i]
            phiyyvals[i,3] = self.tabphi22yy[i,i] * self.c[i+self.nd]

        #do the sum
        xcomp = sum(phivals[:,0]+ phivals[:,1])
        ycomp = sum(phivals[:,2]+ phivals[:,3])

        xcompx = sum(phixvals[:,0]+ phixvals[:,1])
        xcompy = sum(phiyvals[:,0]+ phiyvals[:,1])
        xcompxx = sum(phixxvals[:,0]+ phixxvals[:,1])
        xcompxy = sum(phixyvals[:,0]+ phixyvals[:,1])
        xcompyy = sum(phiyyvals[:,0]+ phiyyvals[:,1])

        ycompx = sum(phixvals[:,2]+ phixvals[:,3])
        ycompy = sum(phiyvals[:,2]+ phiyvals[:,3])
        ycompxx = sum(phixxvals[:,2]+ phixxvals[:,3])
        ycompxy = sum(phixyvals[:,2]+ phixyvals[:,3])
        ycompyy = sum(phiyyvals[:,2]+ phiyyvals[:,3])

        #add back in mean
        xcomp = xcomp + bx0
        ycomp = ycomp + by0

        #scale for a grid of spacing dx
        fdf = 1.0/dx
        sdf = 1.0/(dx*dx)
    
        #these are the derivatives of the x and y vector components
        xcompx = xcompx*fdf
        xcompy = xcompy*fdf
        ycompx = ycompx*fdf
        ycompy = ycompy*fdf

        xcompxx = xcompxx*sdf
        xcompxy = xcompxy*sdf
        xcompyy = xcompyy*sdf
        ycompxx = ycompxx*sdf
        ycompxy = ycompxy*sdf
        ycompyy = ycompyy*sdf

        return [xcompx,xcompy,ycompx,ycompy,xcompxx,xcompxy,xcompyy,ycompxx,ycompxy,ycompyy]

