#This is a simple implementation of e2tc, to be integrated in the simulator

import numpy as np
import matplotlib.pyplot as plt

def best_action(theta):
    #This computes the maximum of x^\top \theta for all x in the set of allowed actions
    return theta/np.linalg.norm(theta)

def e2tc(base,sigma2,theta,T,alpha):
    d = np.size(theta)
    X = np.zeros((T,d))
    Y = np.zeros(T)
    r = np.zeros(T)
    #phase 1: explore base arms in a round robin fashion
    phase_1_end = False
    i=0
    while not phase_1_end:
        a = d*(2 ** i - 1)
        b = d*(2 ** (i+1) -1)
        ni = d * 2 ** (i-1)
        deltai = min(1,d*ni/T)
        Ui = (sigma2*d*d/ni)*(1 + 2*np.sqrt((1/d)*np.log(1/deltai) + (2/d)*np.log(1/deltai)))
        for t in range(a,b):
            X[t,:] = base[t % d,:] 
            Y[t] = np.dot(X[t,:],theta) + np.sqrt(sigma2)*np.random.normal()
        hattheta = np.linalg.inv(  X[a:b,:].transpose() @ X[a:b,:]) @ X[a:b,:].transpose() @ Y[a:b]
        print(hattheta)
        print("Norm",np.linalg.norm(hattheta))
        print("Threshold",alpha*Ui)
        print("Phase ",i,"\n")
        if np.linalg.norm(hattheta) > alpha*Ui:
            phase_1_end = True
            phase_1_end_t = t
        else:
            i += 1
    #phase 2: explore base arms in order to estimate theta
    Ne = d*np.sqrt(sigma2)*np.ceil(np.sqrt(T)/np.linalg.norm(hattheta))
    a = phase_1_end_t + 1
    b = int(a + Ne)
    for t in range(a,b):
        X[t,:] = base[t % d,:] 
        Y[t] = np.dot(X[t,:],theta) + np.sqrt(sigma2)*np.random.normal()
    hattheta = np.linalg.inv(  X[a:b,:].transpose() @ X[a:b,:]) @ X[a:b,:].transpose() @ Y[a:b]
    phase_2_end_t = b
    print(hattheta) 
    print(theta) 
    #phase 3: play greedily until the time horizon runs out
    a = phase_2_end_t
    b = T
    xhat = best_action(hattheta) 
    for t in range(a,b):
        X[t,:] = xhat
        Y[t] = np.dot(X[t,:],theta) + np.sqrt(sigma2)*np.random.normal()
    print("Xhat",xhat)
    print("Xstar",best_action(theta))
    #plot cumulative reward as a function of time
    plt.plot(np.arange(T),np.cumsum(np.ones(T)*np.linalg.norm(theta) - Y))
    plt.show()

#Example of a run
T = 10 ** 6
theta = np.random.rand(4)
base = np.eye(4)
sigma2 = 1
alpha = 1
e2tc(base,sigma2,theta,T,alpha)

