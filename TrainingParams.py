#!/usr/bin/env python
from numpy import *
rCut=6.0
x=arange(0.1,rCut,0.01)
theta=arange(0,pi,pi/100)

def g2(zeta,lamda,eta):
    output = zeros((x.shape[0],theta.shape[0]))
    for i in range(x.shape[0]):
        for j in range(theta.shape[0]):
            output[i,j]=2**(1-zeta)*(1+lamda*cos(theta[j]))**zeta*exp(-eta*x[i]**2)*0.5*(cos(pi*x[i]/Rc)+1)                           
    return output

def g1(eta,rs):
   return exp(-eta*(x-rs)**2)*0.5*(cos(pi*x/Rc)+1)
   
def plotG1Examples(etaMin,RsMin):
    eta=etaMin
    plot(x,g1(eta,RsMin),linewidth=4.0,label='eta=%s'%str(eta))
    eta=etaMin*2
    plot(x,g1(eta,RsMin),linewidth=4.0,label='eta=%s'%str(eta))
    eta=etaMin*4
    plot(x,g1(eta,RsMin),linewidth=4.0,label='eta=%s'%str(eta))
    xlabel(r'$R_(ij)$')
    ylabel(r'$G^1_i$')
    legend(loc='best')
