# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:58:48 2016

@author: Avinash
"""
import numpy
from math import *

def global_var(var_set=0,k1=3.3,N_veh=5):
    if var_set==0:
        global T,N,M,g,l,a
        global Pl,Ph,C,E,ei,ef,s,f,R,B,R_min
        T=24
        M=0.05
        N=N_veh
        data=numpy.genfromtxt('variable_sets\set1\param.csv', delimiter=',')
        g=data[1]
        l=data[2]
        a=data[3]
        Pl=(numpy.ones(shape=(1,N))*2.2)[0]
        Ph=(numpy.ones(shape=(1,N))*k1)[0]
        C=(numpy.ones(shape=(1,N))*16)[0]
        E=(numpy.ones(shape=(1,N))*0.9)[0]
        #ei=(numpy.ones(shape=(1,N))*0.1)[0]
        #R=200
        R=500
        #B=8.5
        #B=10
        B=2.5
        R_min=10        
        ef=(numpy.ones(shape=(1,N))*0.9)[0]
        dataset=numpy.genfromtxt(str(N)+'data.csv', delimiter=',')
        ei=dataset[0]
        s=dataset[1]
        f=dataset[2]
            

#def global_var(var_set=0,k1=3.3,N_veh=5):
#    if var_set==0:
#        global T,N,M,g,l,a
#        global Pl,Ph,C,E,ei,ef,s,f,R,B,R_min
#        T=24
#        M=0.05
#        N=N_veh
#        data=numpy.genfromtxt('variable_sets\set1\param.csv', delimiter=',')
#        g=data[1]
#        l=data[2]
#        a=data[3]
#        Pl=(numpy.ones(shape=(1,N))*2.2)[0]
#        Ph=(numpy.ones(shape=(1,N))*k1)[0]
#        C=(numpy.ones(shape=(1,N))*16)[0]
#        E=(numpy.ones(shape=(1,N))*0.9)[0]
#        #ei=(numpy.ones(shape=(1,N))*0.1)[0]
#        R=200
#        #B=8.5
#        #B=10
#        B=2.5
#        R_min=10
#        if N==5:
#            ei=numpy.array([0.1,0.2,0.1,0.3,0.2])        
#            ef=(numpy.ones(shape=(1,N))*0.9)[0]
#            s=numpy.array([1.482306278,6.63753048,7.075334279,7.724346641,9.667770029])
#            f=numpy.array([21.97939522,21.83438483,18.59006788,21.81806898,18.7517113])        
#        if N==50:
#            ef=(numpy.ones(shape=(1,N))*0.9)[0]
#            dataset=numpy.genfromtxt('50data.csv', delimiter=',')
#            ei=dataset[0]
#            s=dataset[1]
#            f=dataset[2]
            
def variable_gen(N=50):
    ei=numpy.zeros(shape=(1,N))[0]
    s=[]
    for i in range(N):
        ei[i]=numpy.random.randint(1,4)*0.1
        # Vehicle can come upto 18th hour
        s.append(numpy.random.random()*18)
    s.sort()
    s=numpy.array(s)
    f=numpy.zeros(shape=(1,N))[0]
    for i in range(N):
        f[i]=s[i]+1+numpy.random.random()*(5)
    dataset=numpy.zeros(shape=(3,N))
    dataset[0]=ei
    dataset[1]=s
    dataset[2]=f
    numpy.savetxt(str(N)+'data'+".csv",dataset,delimiter=",")    

def static(p,mode=0):
    # mode=0 for LP-R-SCSP 
    # mode=1 for LP-C-SCSP
    #print '1'
    p=p.reshape((T,N))
    u1=10
    u2=10
    u3=10
    h=numpy.zeros(shape=(T,N))
    # Calculation of h[t][i]
    for t in range(T):
        for i in range(N):
            if t>int(s[i]) and t<int(f[i]):
                h[t][i]=1
            elif t==int(s[i]) and s[i]==int(s[i]):
                h[t][i]=0
            elif t==int(s[i]) and s[i]!=int(s[i]):
                h[t][i]=int(s[i]+1)-s[i]
            elif t==int(f[i]) and f[i]!=int(f[i]):
                h[t][i]=f[i]-int(f[i])
            else:
                h[t][i]=0
    #print '2',h.shape
    x=numpy.zeros(shape=(T,N))
    # Calcution of x
    for t in range(T):
        for i in range(N):
            if t==0:
                if t==int(s[i]):
                    x[t][i]=ei[i]   # inintial e
                elif t==int(f[i]+1):
                    x[t][i]=ef[i]   # final e
                else:
                    x[t][i]=0
            else:
                if t==int(s[i]):
                    x[t][i]=ei[i]   # inintial e
                elif t==int(f[i]+1):
                    x[t][i]=ef[i]   # final e
                else:
                    x[t][i]=x[t-1][i]+E[i]*h[t-1][i]*p[t-1][i]/C[i]
    #print '3',x.shape
    P1=numpy.zeros(shape=(T,N))                
    # Calculation of P1(maximum charging rate limit)
    for t in range(T):
        for i in range(N):
            P1[t][i]=(ef[i]-x[t][i])*C[i]/E[i]
    
    u=numpy.zeros(shape=(T,N))
    d=numpy.zeros(shape=(T,N))
    r=numpy.zeros(shape=(T,N))
    # Calculation of u, d and r
    for t in range(T):
        for i in range(N):
            if h[t][i]==1:
                u[t][i]=p[t][i]-Pl[i]  # Pl is lower charging rate limit
                d[t][i]=min(P1[t][i],Ph[i])-p[t][i]  # Ph is higher charging rate limit
            else:
                u[t][i]=0
                d[t][i]=0
            r[t][i]=u[t][i]+d[t][i]
    #print '4'
    if mode==0:
        # Objective function value calculation
        fval1=0
        for t in range(T):
            f1=0
            for i in range(N):
                f1+=r[t][i]
            fval1+=a[t]*f1
        fval2=0
        for t in range(T):
            for i in range(N):
                fval2+=p[t][i]*h[t][i]
        fval=fval1+M*fval2
    elif mode==1:
        fval=0
        for t in range(T):
            k1=0
            for i in range(N):
                k1+=p[t][i]*h[t][i]
            fval+=(M+g[t])*k1        
    ## Penalty calculations
    # Constraint Number 11
    penalty1=0
    for t in range(T):
        c1=0
        psum=0
        dsum=0
        for i in range(N):
            psum+=p[t][i]
            dsum+=d[t][i]
        c1=psum+l[t]+dsum-R
        if c1>0:
            penalty1+=u1*(c1**2)
    penalty2=0
    if mode==0:
        # Constraint number 7
        c2=0
        for t in range(T):
            k1=0
            for i in range(N):
                k1+=p[t][i]*h[t][i]
            c2+=(M+g[t])*k1
        c2=c2-B
        if c2>0:
            penalty2=u2*(c2**2)
    elif mode==1:
        # New constraint for cost minimization
        c2=0
        for t in range(T):
            k1=0
            for i in range(N):
                k1+=r[t][i]
            c2+=a[t]*k1
        k2=0
        for t in range(T):
            for i in range(N):
                k2+=p[t][i]*h[t][i]
        c2=c2+M*k2-R_min
        if c2<0:
            penalty2=u2*(c2**2)
    # Constraint number 6 or boundary constraint
    penalty3=0
    for t in range(T):
        for i in range(N):
            if p[t][i]<Pl[i]:
                penalty3+=u3*(p[t][i]-Pl[i])**2
            if p[t][i]>Ph[i]:
                penalty3+=u3*(Ph[i]-p[t][i])**2 
    #if mode==0: 
     #   fval=fval-penalty1-penalty2-penalty3
      #  return(-fval,fval+penalty1+penalty2+penalty3)
    #elif mode==1:
        #fval=fval+penalty1+penalty3
     #   return(fval-penalty1-penalty3)
    if mode==0:
        return -fval
    else:
        return fval
    
def dynamic(p,x,i,d,mode=0,get_d=0):
    # mode=0 for LP-R-DCSP 
    # mode=1 for LP-C-DCSP
    p[i]=x
    p=p.T
    u1=10
    u2=10
    u3=10
    h=numpy.zeros(shape=(T,N))
    # Calculation of h[t][i]

    for t in range(T):
        if t>int(s[i]) and t<int(f[i]):
            h[t][i]=1
#            if i==1:
#                print ('1',h[t][i])          
        elif t==int(s[i]) and s[i]==int(s[i]):
            h[t][i]=0
#            if i==1:
#                print ('2',h[t][i])
        elif t==int(s[i]) and s[i]!=int(s[i]):
            h[t][i]=int(s[i]+1)-s[i]
#            if i==1:
#                print ('3',h[t][i])
        elif t==int(f[i]) and f[i]!=int(f[i]):
            h[t][i]=f[i]-int(f[i])
#            if i==1:            
#                print ('4',h[t][i])
        else:
            h[t][i]=0
#            if i==1:           
#                print ('5',h[t][i])
    
    
    x=numpy.zeros(shape=(T,N))
    # Calcution of x
    for t in range(T):
        if t==0:
            if t==int(s[i]):
                x[t][i]=ei[i]   # inintial e
            elif t==int(f[i]+1):
                x[t][i]=ef[i]   # final e
            else:
                x[t][i]=0
        else:
            if t==int(s[i]):
                x[t][i]=ei[i]   # inintial e
            elif t==int(f[i]):
                x[t][i]=ef[i]   # final e
            else:
                x[t][i]=x[t-1][i]+E[i]*h[t-1][i]*p[t-1][i]/C[i]
    
    P1=numpy.zeros(shape=(T,N))                
    # Calculation of P1(maximum charging rate limit)
    for t in range(T):
        P1[t][i]=(ef[i]-x[t][i])*C[i]/E[i]
    
    u=numpy.zeros(shape=(T,N))
    #d=numpy.zeros(shape=(T,N))
    r=numpy.zeros(shape=(T,N))
    # Calculation of u, d and r
    for t in range(T):
        if h[t][i]==1:
            u[t][i]=p[t][i]-Pl[i]  # Pl is lower charging rate limit
            d[t][i]=min(P1[t][i],Ph[i])-p[t][i]  # Ph is higher charging rate limit
        else:
            u[t][i]=0
            d[t][i]=0
        r[t][i]=u[t][i]+d[t][i]
    if mode==0:
        # Objective function value calculation
        fval1=0
        fval2=0
#        print h.T[i],r.T[i]
#        print a[t],p.T[i]
        for t in range(T):
            fval1+=a[t]*r[t][i]
            fval2+=p[t][i]*h[t][i]
#        print 'h',h.T[i],'r',r.T[i]
#        print 'a',a[t],'p',p.T[i]
#        print ('1',fval1),('2',fval2)
#        print '1'
#        print h.T[i],r.T[i]
##        print '2'        
#        print a[t],p.T[i]
#        print '3',fval1
#        print '4',fval2
            #if i==1:
                #print fval1,fval2,'a',a[t],'r',r[t][i],'p',p[t][i],'h',h[t][i]
        fval=fval1+M*fval2
    elif mode==1:
        fval=0
        for t in range(T):
            fval+=(M+g[t])*p[t][i]*h[t][i]       
    #print h.T[i]
    ## Penalty calculations
    # Constraint Number 11
    penalty1=0
    for t in range(T):
        c1=0
        psum=0
        dsum=0
        for i_temp in range(i+1):
            psum+=p[t][i_temp]
            dsum+=d[t][i_temp]
        c1=psum+l[t]+dsum-R
        if c1>0:
            penalty1+=u1*(c1**2)
    #print '1 temp'
    penalty2=0
    if mode==0:
        # Constraint number 7
        c2=0
        for t in range(T):
            c2+=(M+g[t])*p[t][i]*h[t][i]
        c2=c2-B
        if c2>0:
            penalty2=u2*(c2**2)        
    # Constraint number 6 or boundary constraint
    penalty3=0
    for t in range(T):
        if p[t][i]<Pl[i]:
            penalty3+=u3*(p[t][i]-Pl[i])**2
        if p[t][i]>Ph[i]:
            penalty3+=u3*(Ph[i]-p[t][i])**2 
    if get_d==1:
        return(d.T[i])
    if mode==0: 
        #print penalty2
        fval=fval-penalty1-penalty2-penalty3
        return(-fval)
    elif mode==1:
        fval=fval+penalty1+penalty3
        return(fval)
