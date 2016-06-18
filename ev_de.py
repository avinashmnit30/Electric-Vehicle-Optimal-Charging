# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 18:01:24 2015

@author: Avinash
"""

import numpy as np 
from numpy import *
import numpy 
from math import *  
import ev_charge_schedule_modification1 as ev
#import ev_charge_schedule.static as func1
#import ev_charge_schedule.dynamic as func2 
import time  
#from numba import double
from numba.decorators import autojit
func1=ev.static

func=autojit(func1)
mode=1
runs=1
maxiter=2000  
F=0.5 # Mutation Factor between 0 to 2
CR=0.2 # Probability 1. Put 0.9 if parameters are dependent while 0.2 if parameters are independent(seperable) 
N=40
D=100*24 # Number of particles
ev.global_var(var_set=0,N_veh=int(D/float(24)))

# boundary constraints
ub=numpy.random.random(size=(1,D))[0]
lb=numpy.random.random(size=(1,D))[0]
i=0
while i<D:
    ub[i]=8.8
    lb[i]=2.2
    i+=1


fitness_val=numpy.zeros(shape=(runs,maxiter))
best_pos=numpy.zeros(shape=(runs,D))
for run_no in range(runs):
    # target vector initializtion
    x=numpy.random.uniform(size=(N,D))
    i=0
    while i<N:
        j=0
        while j<D:
            x[i][j]=lb[j]+x[i][j]*(ub[j]-lb[j])
            j+=1
        i+=1
    
    v=np.zeros_like(x)  # donar vectors
    u=np.zeros_like(x)  # trail vector
    
    g=numpy.zeros(shape=(1,D))[0]  # best vector found so far

    # target vector initial fitness evaluation
    x_fit=numpy.random.uniform(size=(1,N))[0]
    i=0
    while i<N:
        x_fit[i]=func(x[i],mode=mode)
        i+=1
    u_fit=np.zeros_like(x_fit)
    
    j=0
    i=1
    while i<N:
        if x_fit[j]>x_fit[i]:
            j=i
        i+=1
    g_fit=x_fit[j]
    g=x[j].copy()
    
    
    time1=time.time()
    it=0
    while it<maxiter:        
        # Mutation stage
        for i in range(N):
            r1=i
            while r1==i:
                r1=np.random.randint(low=0,high=N)
            r2=i
            while r2==i or r2==r1:
                r2=np.random.randint(low=0,high=N)
            r3=i
            while r3==i or r3==r1 or r3==r2:
                r3=np.random.randint(low=0,high=N)
            v[i]=x[r1]+(x[r2]-x[r3])*F
            for j in range(D):
    #            if v[i][j]>ub[j]:
    #                v[i][j]=v[i][j]-(1+numpy.random.rand())*(v[i][j]-ub[j])
    #            if v[i][j]<lb[j]:
    #                v[i][j]=v[i][j]-(1+numpy.random.rand())*(v[i][j]-lb[j])
    #            if v[i][j]>ub[j]:
    #                v[i][j]=ub[j]
    #            if v[i][j]<lb[j]:
    #                v[i][j]=lb[j]
                if v[i][j]>ub[j]:
                    #v[i][j]=v[i][j]-1.1*(v[i][j]-ub[j])
                    v[i][j]=lb[j]+numpy.random.random()*(ub[j]-lb[j])
                if v[i][j]<lb[j]:
                    v[i][j]=lb[j]+numpy.random.random()*(ub[j]-lb[j])
                    #v[i][j]=v[i][j]-1.1*(v[i][j]-lb[j])
            
        # Recombination stage
        for i in range(N):   
            for j in range(D):
                if np.random.random()<=CR or j==numpy.random.randint(0,D):
                    u[i][j]=v[i][j]
                else:
                    u[i][j]=x[i][j]
        
        # Selection stage
        for i in range(N):
            u_fit[i]=func(u[i],mode=mode)
            if u_fit[i]<x_fit[i]:
                x[i]=u[i].copy()
                x_fit[i]=u_fit[i]
                if u_fit[i]<g_fit:
                    g=u[i].copy()
                    g_fit=u_fit[i]
        fitness_val[run_no][it]=g_fit
        print it,g_fit                
        it+=1
    best_pos[run_no]=g.copy()
    time2=time.time()
    print time2-time1
    run_no+=1
numpy.savetxt("DE_fitness_d1_m2"+str(mode)+str(D)+".csv",fitness_val,delimiter=",")
numpy.savetxt("DE_bestpos_d1_m2"+str(mode)+str(D)+".csv",best_pos,delimiter=",")
