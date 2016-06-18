# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 15:25:30 2015

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
mode=0
D=100*24 # Number of particles
ev.global_var(var_set=0,N_veh=int(D/float(24)))


   
runs=1
maxiter=5
N=500  # Number of particles
N_par=40 # Number of parents
N_mut=40
C_prob=0.8 # Crossover probability
M_prob=0.2 # Mutation probability
b_fac=1.1
cross_type=1

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
    # Genome position initializtion
    x=numpy.random.uniform(size=(N,D))
    i=0
    while i<N:
        j=0
        while j<D:
            x[i][j]=lb[j]+x[i][j]*(ub[j]-lb[j])
            j+=1
        i+=1
        
    
    # Genome initial fitness evaluation
    x_fit=numpy.random.uniform(size=(1,N))[0]
    i=0
    while i<N:
        x_fit[i]=func(x[i],mode=mode)            
        i+=1
    g_fit=x_fit[0]
    g=x[0].copy()
       
    time1=time.time()
    it=0
    while it<maxiter:
        # Natural Selection i.e. sorting population according to fitness
        for j in range(N-1):
            min_i=j
            i=j+1
            while i<N:
                if x_fit[min_i]>x_fit[i]:
                    min_i=i
                i+=1  
            if min_i!=j:
                x_fit[min_i],x_fit[j]=x_fit[j],x_fit[min_i]
                x[min_i],x[j]=x[j].copy(),x[min_i].copy() 
        # Mating/Crossover
        i=N_par+N_mut
        while i<N:
            p1=numpy.random.randint(0,N_par)
            p2=p1
            while p2==p1:
                p2=numpy.random.randint(0,N_par)
            for j in range(D):
                if numpy.random.random()<C_prob:
                    if cross_type==0:    
                        b1=numpy.random.random()*b_fac
                        b2=numpy.random.random()*b_fac
                        x[i][j]=b1*x[p1][j]+(1-b1)*x[p2][j]
                        x[i+1][j]=b2*x[p1][j]+(1-b2)*x[p2][j]
                        if x[i][j]>ub[j]:
                            x[i][j]=lb[j]+numpy.random.random()*(ub[j]-lb[j])
                        if x[i][j]<lb[j]:
                            x[i][j]=lb[j]+numpy.random.random()*(ub[j]-lb[j])
                    elif cross_type==1:
                        b=numpy.random.random()*b_fac
                        x[i][j]=x[p1][j]-b*(x[p1][j]-x[p2][j])
                        x[i+1][j]=x[p2][j]+b*(x[p2][j]-x[p1][j])
                        if x[i][j]>ub[j]:
                            x[i][j]=lb[j]+numpy.random.random()*(ub[j]-lb[j])
                        if x[i][j]<lb[j]:
                            x[i][j]=lb[j]+numpy.random.random()*(ub[j]-lb[j])           
                else:
                    x[i][j]=x[p1][j]
                    x[i+1][j]=x[p2][j]
            
            i+=2
        # Mutation 
        i=N_par
        while i<N_par+N_mut:
            for j in range(D):
                if numpy.random.random()<M_prob:
                    if numpy.random.random()<0.5:
                        x[i][j]=x[i-N_par][j]+(lb[j]+numpy.random.random()*(ub[j]-lb[j]))/(lb[j]+ub[j])
                    else:
                        x[i][j]=x[i-N_par][j]-(lb[j]+numpy.random.random()*(ub[j]-lb[j]))/(lb[j]+ub[j])
                else:
                    x[i][j]=x[i-N_par][j]
            i+=1
#        for i in range(N):
#            for j in range(D):
#                if numpy.random.random()<M_prob:
#                    if numpy.random.random()<0.5:
#                        x[i][j]+=(lb[j]+numpy.random.random()*(ub[j]-lb[j]))/(lb[j]+ub[j])
#                    else:
#                        x[i][j]-=(lb[j]+numpy.random.random()*(ub[j]-lb[j]))/(lb[j]+ub[j])
#        for i in range(N_par):
#            for j in range(D):
#                if numpy.random.random()<M_prob:
#                    if numpy.random.random()<0.5:
#                        x_temp[j]=x[i][j]+(lb[j]+numpy.random.random()*(ub[j]-lb[j]))/(lb[j]+ub[j])
#                    else:
#                        x_temp[j]=x[i][j]-(lb[j]+numpy.random.random()*(ub[j]-lb[j]))/(lb[j]+ub[j])
#                else:
#                    x_temp[j]=x[i][j]
#            x_temp_fit=func(x[i],D,battery_no=battery_no,state=state,Cr=CDr,V_m=V_m,SOC_m=SOC_m,lb=lb[0],ub=ub[0])              
#            if x_temp_fit<x_fit[i]:
#                x[i]=x_temp.copy()
#                x_fit[i]=x_temp_fit
        for i in range(N):
            x_fit[i]=func(x[i],mode=mode)            
                    
            if x_fit[i]<g_fit:
                g_fit=x_fit[i]
                g=x[i].copy()
        fitness_val[run_no][it]=g_fit
        print it,g_fit                
        it+=1
    print run_no
    best_pos[run_no]=g.copy()
    time2=time.time()
    print time2-time1
    run_no+=1
numpy.savetxt("GA_fitness_d1_m2"+str(mode)+str(D)+".csv",fitness_val,delimiter=",")
numpy.savetxt("GA_bestpos_d1_m2"+str(mode)+str(D)+".csv",best_pos,delimiter=",")
               
