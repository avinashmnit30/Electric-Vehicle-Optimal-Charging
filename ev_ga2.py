# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 15:25:30 2015

@author: Avinash
"""

import numpy as np 
from numpy import *
import numpy 
from math import *  
import ev_charge_schedule as ev
#import ev_charge_schedule.static as func1
#import ev_charge_schedule.dynamic as func2 
import time  
#from numba import double
from numba.decorators import autojit
func1=ev.dynamic
func=autojit(func1)
mode=1
runs=1
F=0.8 # Mutation Factor between 0 to 2
CR=0.9 # Probability 1. Put 0.9 if parameters are dependent while 0.2 if parameters are independent(seperable) 
D=24 
N_veh=200
   

maxiter=10
N=1000  # Number of particles
N_par=40 # Number of parents
C_prob=0.3 # Crossover probability
M_prob=0.2 # Mutation probability
b_fac=1.1

value=numpy.zeros(shape=(1,N_veh))
counterk1=0
for k1 in [8.8]:
    d=numpy.zeros(shape=(N_veh,24))
    p_sol=numpy.zeros(shape=(N_veh,24))
    ev.global_var(var_set=0,k1=k1,N_veh=N_veh)
    for veh_no in range(N_veh):
        # boundary constraints
        ub=numpy.random.random(size=(1,D))[0]
        lb=numpy.random.random(size=(1,D))[0]
        i=0
        while i<D:
            ub[i]=k1
            lb[i]=2.2
            i+=1  
   



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
                x_fit[i],fval,penalty1,penalty2,penalty3=func(p_sol,x[i],veh_no,d.T,mode=mode)
                i+=1
            g_fit=x_fit[0]
            g=x[0].copy()
               
        #    maxiter=1
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
                i=N_par
                while i<N:
                    p1=numpy.random.randint(0,N_par)
                    p2=p1
                    while p2==p1:
                        p2=numpy.random.randint(0,N_par)
                    for j in range(D):
        #                if numpy.random.random()<C_prob:
        #                    b1=numpy.random.random()*b_fac
        #                    b2=numpy.random.random()*b_fac
        #                    x[i][j]=b1*x[p1][j]+(1-b1)*x[p2][j]
        #                    x[i+1][j]=b2*x[p1][j]+(1-b2)*x[p2][j]
        #                    if x[i][j]>ub[j]:
        #                        x[i][j]=x[i][j]-1.1*(x[i][j]-ub[j])
        #                    if x[i][j]<lb[j]:
        #                        x[i][j]=x[i][j]-1.1*(x[i][j]-lb[j])
                        if numpy.random.random()<C_prob:
                            b=numpy.random.random()*b_fac
                            x[i][j]=x[p1][j]-b*(x[p1][j]-x[p2][j])
                            x[i+1][j]=x[p2][j]+b*(x[p2][j]-x[p1][j])
                            if x[i][j]>ub[j]:
                                x[i][j]=x[i][j]-1.1*(x[i][j]-ub[j])
                            if x[i][j]<lb[j]:
                                x[i][j]=x[i][j]-1.1*(x[i][j]-lb[j])
                    
                        else:
                            x[i][j]=x[p1][j]
                            x[i+1][j]=x[p2][j]
                    
                    i+=2
                # Mutation 
                for i in range(N):
                    for j in range(D):
                        if numpy.random.random()<M_prob:
                            x[i][j]=lb[j]+numpy.random.random()*(ub[j]-lb[j])
                            
                for i in range(N):
                    x_fit[i],fval,penalty1,penalty2,penalty3=func(p_sol,x[i],veh_no,d.T,mode=mode)
    
                    if x_fit[i]<g_fit:
                        g_fit=x_fit[i]
                        g=x[i].copy()
                print it,g_fit                
                it+=1
            time2=time.time()
            #print time2-time1
            run_no+=1
     
            d_temp=func(p_sol,g,veh_no,d.T,mode=mode,get_d=1)
            d[veh_no]=d_temp
            p_sol[veh_no]=g
            value[counterk1][veh_no]=fval
    counterk1+=1
        
        
        
numpy.savetxt('GA'+str(N_veh)+"dynamicev"+'mode'+str(mode)+".csv",value,delimiter=",")
