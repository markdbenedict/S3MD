#!/usr/bin/env python
from numpy import *
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from mpl_toolkits.mplot3d import Axes3D

rCut=6.0
x=arange(0.1,rCut,0.1)
theta=arange(0,2*pi+pi/100,pi/100)

def g2(zeta,lamda,eta):
    R,P=meshgrid(x,theta)
    #output = zeros((x.size,theta.size))
    #for i in range(x.size):
    #    for j in range(theta.size):
    #(1+lamda*cos(P))**zeta
    output=2**(1-zeta)*exp(-eta*R**2)*0.5*(cos(pi*R/rCut)+1)                           
    X,Y=R*cos(P),R*sin(P)
    return [X,Y,output]

def g1(eta,rs):
   return exp(-eta*(x-rs)**2)*0.5*(cos(pi*x/rCut)+1)
   
def plotG1Examples(etaRange,RsRange):
    #plot all combos of eta and Rs
    for eta in etaRange:
        for Rs in RsRange:
            plt.plot(x,g1(eta,Rs),linewidth=4.0,label='eta=%s Rs=%s'%(str(eta),str(Rs)))
   
    plt.xlabel(r'$R_(ij)$')
    plt.ylabel(r'$G^1_i$')
    plt.legend(loc='best')

def plotG1pairs(listOfParams):
    #plot all combos of eta and Rs
    for eta,Rs in listOfParams:
            plt.plot(x,g1(eta,Rs),linewidth=4.0,label='eta=%s Rs=%s'%(str(eta),str(Rs)))
   
    plt.xlabel(r'$R_(ij)$')
    plt.ylabel(r'$G^1_i$')
    plt.legend(loc='best')
    
def plotG2(zeta,lambd,eta):
    #plot all combos of eta and Rs
    fig=plt.figure()
    ax=Axes3D(fig)
    X,Y,Z=g2(zeta, lambd,eta)
    surf=ax.plot_surface(X,Y,Z,rstride=4,cstride=4,cmap=cm.jet)
    #ax.set_ylabel(r'$\theta$')
    #ax.set_xlabel(r'$R_(ij)$')
    ax.set_zlabel(r'$G^2_i$')
    plt.title('$G^2_i for \lambda=%3.2f,\zeta=%3.2f,\eta=%3.2f'%(lambd,zeta,eta))
    fig.colorbar(surf,shrink=0.5,aspect=5)
    plt.show()