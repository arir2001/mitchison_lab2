#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 16:35:38 2022

@author: annamarierooney
"""
print('Name: Anna Rooney')
print('Student Number: 19333456', '\n')

import numpy as np
import matplotlib.pylab as plt
import math as math
import numpy.random as ran
import statistics as st

ran.seed(403)
#%%
'''part one'''
print('\n',"part one",'\n')
def fact(n):
    fact = 1
    for i in range(1, n+1):
        fact = fact * i 
    return fact


def fish(mean, n):
    a = mean**n
    b = fact(int(n))
    c = np.exp(-mean)
    return (a/b)*c

values = np.linspace(0,100,101)

for x in [1, 5, 10]:
    print('the poisson for n = 20 at mean =', x, 'is', fish(x,20))
    print('the standard dev for n=20 at mean=,', x, 'is ', np.sqrt(x))

plt.figure(1)
plt.plot(values,[fish(1, n) for n in values],label='Mean is 1')
plt.plot(values,[fish(5, n) for n in values],label='Mean is 5')
plt.plot(values,[fish(10, n) for n in values],label='Meanis 10')
plt.title('Poisson Distributions at n=20, with means 1,5,10')
plt.xlabel('n')
plt.ylabel('P(n)')
plt.xlim(0, 35)
plt.legend()
plt.show()
#plt.savefig('poisson1.png', bbox_inches='tight')


#%%
'''part two'''
print('\n',"part two",'\n')
#define the functions
def sumP(mean, N):
    end = 0
    for i in range(0, N+1): #sum over from n = 0 to n= N
        end += fish(mean, i)
    return end

def nP(mean, N):
    return N*sumP(N, mean)
    
def n2P(mean, N):
    return (N**2)*sumP(mean, N)

gg=[]
zz=[]
#test for normalisation
for g in [1, 5, 10]: #mean
    zz=[]
    gg = []
    for z in range(10, 60, 2): #N
       # print('For mean='+str(g)+'at N ='+str(z)+'the sum is',sumP(g, z))
        zz.append(sumP(g,z))
        gg.append(z)
        if sumP(g, z) == 1.0:
            print('For mean = '+str(g)+' at N ='+str(z)+' the sum is normalised to',sumP(g,z))
            print('Hence, at a mean of '+str(g)+' for the P(n) to be normalised, N must be more than or equal to '+str(z))
            break
    plt.figure(2)
    plt.plot(gg, zz, label = '$\mu$ is %i'%g)
    plt.title('Sum of Poisson Distributions')
    plt.xlabel('N')
    plt.ylabel('$\sum^N_{n=0} P(n)$')
    plt.xlim(0, 30)
    plt.legend()
    #plt.savefig('poisson2.png', bbox_inches='tight')

print("At mean = 10, the sum reaches",sumP(g,z),'but never reaches 1(at least with the ranges given by the lab)')
#standard deviation


#%%
'''part three'''
print('\n',"part three",'\n')
N = 50 #number of darts
L = 100 #number of regions

#finds the random distribution
#takes in N, number of darts, and L, number of regions to count up to
def dist(N, L): 
    B = np.zeros(L)
    for n in range(N):
        darts = ran.randint(0,L) #each time this is run, a new random array will be made
        B[darts]+= 1
    return B

BB  = dist(N, L)

plt.figure(800)
plt.hist(BB, color = 'pink')
plt.title('Histogram of darts for L=100, N=50')
plt.xlabel('Darts recieved per region')
plt.ylabel('# regions')
plt.show()

#count thefrequency of darts per region
def counting(B, N):
    c = np.zeros(N) #number of darts possible
    for n in range(N):
        if n in B: #if n is in the array B (which couonted the darts per region), count the frequenccy to the same index
            c[n]+=np.count_nonzero(B == n)
    return c
    
#make H1n, H2n
H1n = counting(dist(N,L),N)
H2n = counting(dist(N,L),N)

print('\n', 'H_1(n)', H1n,'H_2(n)', H2n)
print('\n', 'The sum of H1n and H2n is', H1n+H2n)


T = 10

#return H(n)i,  sum(H(n)), or all Hni for number of trials = T
def trials(N, L, T, horhn, tit):
    H = [np.zeros(L)]*T
    for t in range(T):
        H[t] = counting(dist(N, L), N) #trials t
    Hn = sum(H) #sum of trials
    if horhn == "Hn":
        return Hn      
    if horhn == "H":
        return H[tit]
    if horhn == "all H":
        return H
    
#sumof all the Hn for 10trials
Hn = trials(N, L, T, "Hn", 0)

#normalise and find probability 
def normalised(Hn, L, T):
    Psimn = Hn /(L*T)
    return Psimn

#Normalise the data:
normalHN = normalised(Hn, L, T)

print("Hn is ",Hn)
print('the normalised distribution is', normalHN)
print('The sum of the normalised distribution is', sum(normalHN))


print('the mean number of darts per region is', N/L)
N = 50 #number of darts
L = 100 #number of regions
f = [fish(N/L, j) for j in  np.arange(0,len(normalHN),1)]

plt.figure(3)
plt.plot(f,color='blue',label='Poisson Distribution') 
plt.scatter(np.arange(0,len(normalHN),1), normalHN, color = 'black', label='$P_{sim}(n)$ ')
plt.title('$P_{sim}(n)$ for 10 trials compared to the Poisson Distribution ')
plt.xlabel('L regions with n darts (where L=100, mean = 0.5)')
plt.ylabel('Probability of a region having n darts, N=50')
plt.legend()
#plt.savefig('poisson3.png', bbox_inches='tight')


#%%
'''part four'''

plt.figure(4)
plt.yscale('log')
plt.plot(np.arange(0,len(normalHN),1), normalHN, color = 'green')
plt.plot(f,color='blue',label='Poisson Distribution')
plt.scatter(np.arange(0,len(normalHN),1), normalHN, color = 'black', label='$P_{sim}(n)$ ')

plt.title('$P_{sim}(n)$ for 10 trials compared to the Poisson Distribution ')
plt.xlabel('L regions with n darts (where L=n)')
plt.ylabel('Probability of a region having n darts')
plt.legend()
#plt.savefig('loglog4.png', bbox_inches='tight')


plt.figure(5)
plt.yscale('log')
plt.plot(np.arange(0,len(normalHN),1), normalHN, color = 'green',  label='$P_{sim}(n)$ ')
plt.plot(f,color='blue',label='Poisson Distribution')
plt.scatter(np.arange(0,len(normalHN),1), normalHN, color = 'black', label='$P_{sim}(n)$ ')
#plt.ylim(10e-8, 10**1)
plt.xlim(-1, 10)

plt.title('$P_{sim}(n)$ for 10 trials compared to the Poisson Distribution ')
plt.xlabel('L regions with n darts (where L=n)')
plt.ylabel('Probability of a region having n darts')
plt.legend()
#plt.savefig('loglog4.png', bbox_inches='tight')

#%%
'''part five'''
N = 50 #number of darts
L = 100 #number of regions

Tl = [100, 1000, 10000]
nonzero = [[]]*3
alll = [[]]*3
for g in range(3):
    T = Tl[g]
    D = np.zeros(T)
    kk= trials(N, L, T, "Hn", 0)
    alll[g].append(kk)
    
'''
    for x in range(L):
        D = trials(N, L, T, "H", x) 
        #print(H)
        for k in range(N):
            if D[k] != 0:
                nonzero[g].append(D[k])
                #print(H[k])
    print(nonzero[g])
'''
#%%
a = alll[1]
HNone = normalised(a[0],100, 100)
HNtwo =normalised(a[1], 100, 1000)
HNthree= normalised(a[2],100, 10000)

#normal scale 
plt.figure(6)
#plt.plot(values,[fish(1, n) for n in values],label='Mean is 1')
plt.plot(f,color='black',label='Poisson Distribution', linewidth= 0.5)
plt.scatter(np.arange(0,len(HNone),1), HNone, marker ='*',  label='T=100')
plt.scatter(np.arange(0,len(HNtwo),1), HNtwo, marker = 'o',  label='T=1000')
plt.scatter(np.arange(0,len(HNthree),1), HNthree, marker = '4', label='T=1000')
plt.title('$P_{sim}(n)$ for for 3 different trial lengths, vs Poisson Distribution ,at L=100,  N=50')
plt.xlabel('L regions with n darts (where L=n)')
plt.ylabel('Probability of a region having n darts')
plt.legend()
#plt.savefig('normal5.png', bbox_inches='tight')

#log log scale
plt.figure(7)
plt.yscale('log')
plt.plot(f,color='black',label='Poisson Distribution', linewidth= 0.5)
plt.scatter(np.arange(0,len(HNone),1), HNone, marker ='8', color = 'red', label='T=100')
plt.scatter(np.arange(0,len(HNtwo),1), HNtwo, marker = 'd',label='T=1000')
plt.scatter(np.arange(0,len(HNthree),1), HNthree, marker = '*',label='T=10000')
#plt.plot(np.arange(0,len(HNone),1), HNone, color ='blue', linewidth = 0.5)
#plt.plot(np.arange(0,len(HNtwo),1), HNtwo, color='orange', linewidth = 0.5)
#plt.plot(np.arange(0,len(HNthree),1), HNthree, color = 'green', linewidth = 0.5)
plt.title(' log-log scale: $P_{sim}(n)$ vs Poisson, at L=100,  N=50')
plt.xlabel('L regions with n darts (where L=n)')
plt.ylabel('Probability of a region having n darts')
plt.ylim(10**-7, 10**0)
plt.xlim(-1, 10)
plt.legend()
#plt.savefig('loglog5.png', bbox_inches='tight')

#%%
li= [HNone,HNtwo ,HNthree] #list of the normalised lists
listen = [] #empty list to append elements
listenn = [] #empty list to append elements

#finds the indexof the nonzero elements inthe normalised lists
small = [np.nonzero(HNone), np.nonzero(HNtwo), np.nonzero(HNthree)]

for s in range(3):
    inds = small[s] #choose the indexes 
    co= li[s] #chosoe the corresponding list
    T = Tl[s]
    for ss in range(len(inds)):
        indd = inds[ss] #choose the index from the index list
        listenn.append(co[indd])
        for sss in range(len(indd)):
            ind = indd[sss]
            #print(ind, co[ind]) #find the element with the index
            listen.append(co[ind]) #append the elemnt 
    print('The smallest possible $P_n$ for %i trials is '%T, min(listenn[s]))


#%%
'''part six'''
    
L = 5
N=  50
Tl = [10, 1000, 10000]
nootzero = [[]]*3
ali = [[]]*3
for g in range(3):
    T = Tl[g]
    A = np.shape(T)
    ali[g].append(trials(N, L, T, "Hn", 0))
    '''
    for x in range(L):
        A = trials(N, L, T, "H", x) 
        for k in range(N):
            if A[k] != 0:
                nootzero[g].append(A[k])
    print(nootzero[g])
'''
#%%
'''
a = ali[1]
b= ali[2]

HNone =  17*normalised(a[0],100, 100)
HNtwo =  17*normalised(a[1], 100, 1000)
HNthree=  17*normalised(a[2],100, 10000)

L, N = 5, 50


#normal scale 
plt.figure(11)
plt.plot(values,[fish(10, n) for n in values],label='Mean is 10')
plt.scatter(np.arange(0,len(HNone),1), HNone, marker ='*', label ='T=100')
plt.scatter(np.arange(0,len(HNtwo),1), HNtwo, marker = 'o', label ='T=100')
plt.scatter(np.arange(0,len(HNthree),1), HNthree, marker = '4', label ='T=100')
plt.title('$P_{sim}(n)$ for %i trials compared to the Poisson Distribution '%T)
plt.xlabel('L regions with n darts (where L=n)')
plt.ylabel('Probability of a region having n darts')
plt.legend()
plt.savefig('normal.png', bbox_inches='tight')

#log log scale
plt.figure(12)
plt.yscale('log')
plt.plot(values,[fish(10, n) for n in values],color='black',label='Poisson Distribution', linewidth= 0.5)
plt.scatter(np.arange(0,len(HNone),1), HNone, marker ='*',label='T=100')
plt.scatter(np.arange(0,len(HNtwo),1), HNtwo, marker = 'o',label='T=1000')
plt.scatter(np.arange(0,len(HNthree),1), HNthree, marker = '4',label='T=10000')

plt.title('$P_{sim}(n)$ for different trials, compared to the Poisson Distribution ')
plt.xlabel('L regions with n darts (where L=n)')
plt.ylabel('Probability of a region having n darts')
plt.ylim(10**-5, 10**-0.5)
plt.xlim(-1, 35)
plt.legend()   
'''
     
#%%
b = ali[1]

HNone = normalised(b[0],5, 10)
HNtwo =normalised(b[1], 5, 1000)
HNthree= normalised(b[2],5, 10000)

ff = [fish(N/L, j) for j in  np.arange(0,len(HNtwo),1)]

 #normal scale 
plt.figure(8)
plt.plot(ff,color='black',label='Poisson Distribution', linewidth= 0.5)
plt.scatter(np.arange(0,len(HNone),1), HNone, marker ='*',  label='T=10')
plt.scatter(np.arange(0,len(HNtwo),1), HNtwo, marker = 'o',  label='T=1000')
plt.scatter(np.arange(0,len(HNthree),1), HNthree, marker = '4', label='T=10000')
plt.title('$P_{sim}(n)$ for for 3 different trial lengths, compared to the Poisson Distribution ')
plt.xlabel('L regions with n darts (where L=5)')
plt.ylabel('Probability of a region having n darts, N=50')
plt.legend()
#plt.savefig('normal6.png', bbox_inches='tight')

#log log scale
plt.figure(9)
plt.yscale('log')
plt.plot(ff,color='black',label='Poisson Distribution', linewidth= 0.5)
plt.scatter(np.arange(0,len(HNone),1), HNone, marker ='8', color = 'red', label='T=100')
plt.scatter(np.arange(0,len(HNtwo),1), HNtwo, marker = 'd',label='T=1000')
plt.scatter(np.arange(0,len(HNthree),1), HNthree, marker = '*',label='T=10000')

plt.title(' log-log scale: $P_{sim}(n)$ for 3 different trial lengths, compared to the Poisson Distribution ')
plt.xlabel('L regions with n darts (where L=5)')
plt.ylabel('Probability of a region having n darts, N=50')
plt.ylim(10**-7, 10**0)
plt.xlim(-1, 20)
plt.legend()   
#plt.savefig('loglog6.png', bbox_inches='tight')
