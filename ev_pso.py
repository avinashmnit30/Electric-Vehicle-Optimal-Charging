# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 09:50:39 2015

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
N=40
D=5*24 # Number of particles
ev.global_var(var_set=0,N_veh=int(D/float(24)))

# boundary constraints
ub=numpy.random.random(size=(1,D))[0]
lb=numpy.random.random(size=(1,D))[0]
i=0
while i<D:
    ub[i]=8.8
    lb[i]=2.2
    i+=1


wi=0.9   #initial inertia weight 
wf=0.5  #final inertia weight

fitness_val=numpy.zeros(shape=(runs,maxiter))
best_pos=numpy.zeros(shape=(runs,D))


for run_no in range(runs):
    #particle position-velocity initializtion
    x=numpy.random.uniform(size=(N,D))
    i=0
    while i<N:
        j=0
        while j<D:
            x[i][j]=lb[j]+x[i][j]*(ub[j]-lb[j])
            j+=1
        i+=1
    
    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # particle position found by particle
    
    g=numpy.zeros(shape=(1,D))[0]  # best swarm position found so far
    
    
    #Particle initial fitness evaluation
    x_fit=numpy.random.uniform(size=(1,N))[0]
    i=0
    while i<N:
        x_fit[i]=func(x[i],mode=mode) 
        i+=1
    p_fit=x_fit.copy()
    
    
    j=0
    i=1
    while i<N:
        if p_fit[j]>p_fit[i]:
            j=i
        i+=1
    g_fit=p_fit[j]
    g=p[j].copy()
        
    
    time1=time.time()
    it=0
    while it<maxiter:                  
        w=((wi-wf)*(maxiter-it)/maxiter)+wf #time varying inertia weight
        for i in range(N):
            for j in range(D):
                v[i][j]=w*v[i][j]+2*np.random.rand()*(p[i][j]-x[i][j])+2*np.random.rand()*(g[j]-x[i][j])    
                x[i][j]=x[i][j]+v[i][j]
    #            if x[i][j]>ub[j]:
    #                x[i][j]=ub[j]-(numpy.random.random()/8)*(ub[j]-lb[j])
    #            if x[i][j]<lb[j]:
    #                x[i][j]=lb[j]+(numpy.random.random()/8)*(ub[j]-lb[j])
#                if x[i][j]>ub[j]:
#                    x[i][j]=x[i][j]-1.1*(x[i][j]-ub[j])
#                if x[i][j]<lb[j]:
#                    x[i][j]=x[i][j]-1.1*(x[i][j]-lb[j])
                if x[i][j]>ub[j]:
                    #v[i][j]=v[i][j]-1.1*(v[i][j]-ub[j])
                    x[i][j]=lb[j]+numpy.random.random()*(ub[j]-lb[j])
                if v[i][j]<lb[j]:
                    x[i][j]=lb[j]+numpy.random.random()*(ub[j]-lb[j])
    
        for i in range(N):
            x_fit[i]=func(x[i],mode=mode)            
            if x_fit[i]<p_fit[i]:
                p_fit[i]=x_fit[i]
                p[i]=x[i].copy()
                if p_fit[i]<g_fit:
                    g_fit=p_fit[i]
                    g=p[i].copy()
        fitness_val[run_no][it]=g_fit
        print it,g_fit                
        it+=1
    best_pos[run_no]=g.copy()
    time2=time.time()
    print time2-time1
    run_no+=1
numpy.savetxt("PSO_fitness_d1_m2"+str(mode)+str(D)+".csv",fitness_val,delimiter=",")
numpy.savetxt("PSO_bestpos_d1_m2"+str(mode)+str(D)+".csv",best_pos,delimiter=",")
