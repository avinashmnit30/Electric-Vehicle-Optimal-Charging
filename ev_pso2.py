# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 09:50:39 2015

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
maxiter=500
F=0.8 # Mutation Factor between 0 to 2
CR=0.9 # Probability 1. Put 0.9 if parameters are dependent while 0.2 if parameters are independent(seperable) 
N=40
D=24 
N_veh=200

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
   

        wi=0.9   #initial inertia weight 
        wf=0.5  #final inertia weight
        
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
                x_fit[i],fval,penalty1,penalty2,penalty3=func(p_sol,x[i],veh_no,d.T,mode=mode)
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
                        if x[i][j]>ub[j]:
                            x[i][j]=x[i][j]-1.1*(x[i][j]-ub[j])
                        if x[i][j]<lb[j]:
                            x[i][j]=x[i][j]-1.1*(x[i][j]-lb[j])
            
                for i in range(N):
                    x_fit[i],fval,penalty1,penalty2,penalty3=func(p_sol,x[i],veh_no,d.T,mode=mode)
                    if x_fit[i]<p_fit[i]:
                        p_fit[i]=x_fit[i]
                        p[i]=x[i].copy()
                        if p_fit[i]<g_fit:
                            g_fit=p_fit[i]
                            g=p[i].copy()
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
        
        
        
numpy.savetxt('PSO'+str(N_veh)+"dynamicev"+'mode'+str(mode)+".csv",value,delimiter=",")

