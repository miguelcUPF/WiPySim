#This is a simple implementation of OSUB, to be integrated in the simulator
#Note: we use a simplified algorithm compared to the article https://arxiv.org/abs/1405.5096 where we do not force exploration of the leader

import numpy as np
import matplotlib.pyplot as plt

def get_neighbours_line_graph(k,K):
    #return the neigbours of arm k, inclduing arm k itself (here we consider the line graph), this should be modified for other graphs, of course
    if k==0:
        return [0,1]
    elif k==(K-1):
        return [K-1,K-2]
    else:
        return [k-1,k,k+1]

def kl_ucb_index(m,n,t):
    verbose = False
    #find the KL-UCB index which is the unique solution 1 ge q ge m to n D( m , q ) = log(t) 
    if n== 0:
        return 1
    elif m == 1:
        return 1
    elif m == 0: #in this case D( m, q) = log(1/(1-q)) so that the solution is q =1 t^(-1/n) 
        return 1 - t ** (-1/n) 
    else: #in this case we can use newton's method for the function f(q)= n D(m,q) - log(t), with derivative equal to n ( -m/q  + (1-m)/(1-q) )
        q = m
        #find an initial value for q making sure that f is greater than 0
        while n*( m*np.log(m/q) + (1-m)*np.log((1-m)/(1-q))) - np.log(t) < 0: q = (q+1)/2 
        #apply Newton's method to f 
        f = n*( m*np.log(m/q) + (1-m)*np.log((1-m)/(1-q))) - np.log(t) #function value
        while np.abs(f) > 10 ** (-5):
            f = n*( m*np.log(m/q) + (1-m)*np.log((1-m)/(1-q))) - np.log(t) #function value
            fprime = n*(-m/q + (1-m)/(1-q) ) #function derivative
            q = q - f/fprime #newton step
            #display debugging information at each iteration
            if verbose: print("q",q,"function value",f,"function derivative",fprime)
        return q

def osub(mu,T):
    #simulate one run of OSUB where the expected rewards are Bernoulli, with means given by mu
    verbose = False
    K = np.size(mu)
    rewards = np.zeros(K)
    plays = np.zeros(K)
    muhat = np.zeros(K)
    for t in range(T):
        #compute empirical means
        muhat = rewards/np.maximum(plays,np.ones(K)) 
        #find the leader
        ell = np.argmax(muhat)
        #find the set of arms within distance 1 of the leader
        N = get_neighbours_line_graph(ell,K)
        #compute the KL-UCB index of arms in N
        B = [kl_ucb_index(muhat[k],plays[k],t) for k in N]
        kt = N[np.argmax(B)]
        #select arm and update statistics
        plays[kt] +=1
        rewards[kt] += (np.random.rand() <= mu[kt])
        #display debugging information at each iteration
        if verbose: 
            print('t = ',t)
            print("Plays = ", plays)
            print("Rewards = ", rewards)
            print("Mu hat = ", muhat)
            print("Leader = ", ell)
            print("Leader neigbours = ", N)
            print("Leader neigbours indexes = ", B)
            print("Chosen arm = ", kt)
    return(plays)
#Example of a run on a simple unimodal function: we plot the reward function mu, which is unimodal with respect to the line graph, and the frequency that each arm has been played
K = 50
T = 100000
mu = [1 - 2*(i/K - 0.6) ** 2 for i in range(K)]
plt.plot(range(K),osub(mu,T)/T,3,label='toto')
plt.plot(range(K),mu)
plt.xlabel('Arm index')
plt.ylim([0,1])
plt.ylabel('Rewards and frequency of play')
plt.show()
