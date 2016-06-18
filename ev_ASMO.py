# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 09:02:23 2015

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
D=100*24 # Number of particles
ev.global_var(var_set=0,N_veh=int(D/float(24)))


fitness_val=numpy.zeros(shape=(runs,maxiter))
best_pos=numpy.zeros(shape=(runs,D))

MG=4
prmn=0.5
prmx=0.5

def SMO_a1(SM,SMval,SML,SMU,D,LL,LLval,pr,SM2,SMval2,LLval2,LLL,prob,ub,lb,GL,SMLP,SMUP):
    SMnum=SMLP
    while SMnum<SMUP:
#        print SMnum
        j=0
        while j<D:
            if numpy.random.random()>=pr:
                r1=SMnum
                while r1==SMnum:
                    r1=numpy.random.randint(SML,SMU)
                SM[SMnum][j]=SM[SMnum][j]+numpy.random.random()*(LL[j]-SM[SMnum][j])+((2*numpy.random.random())-1)*(SM[r1][j]-SM[SMnum][j])
                if SM[SMnum][j]>ub[j][0]:
                    SM[SMnum][j]=lb[j][0]+numpy.random.random()*(ub[j][0]-lb[j][0])    
                if SM[SMnum][j]<lb[j][0]:
                    SM[SMnum][j]=lb[j][0]+numpy.random.random()*(ub[j][0]-lb[j][0])    
            j+=1
        SMnum+=1
    SMnum=SMLP
    while SMnum<SMUP:
        SMval[SMnum][0]=func(SM[SMnum],mode=mode)
        if SMval2[SMnum][0]<SMval[SMnum][0]:
            SMval[SMnum][0]=SMval2[SMnum][0]
            SM[SMnum]=SM2[SMnum]
        else:
            SMval2[SMnum][0]=SMval[SMnum][0]
            SM2[SMnum]=SM[SMnum]
            if SMval[SMnum][0]<LLval:
                LLval=SMval[SMnum][0]
                LL=SM[SMnum]
        SMnum+=1
    return(SM[SMLP:SMUP],SMval[SMLP:SMUP],LL,LLval,SM2[SMLP:SMUP],SMval2[SMLP:SMUP])

def SMO_a2(SM,SMval,SML,SMU,D,LL,LLval,pr,SM2,SMval2,LLval2,LLL,prob,ub,lb,GL,SMLP,SMUP):
    SMnum=SMLP  
    while SMnum<SMUP:
        prob[SMnum][0]=0.1+0.9*(LLval/SMval[SMnum][0])
        prob[SMnum][0]=0.5
        SMnum+=1
    count=SMLP
    while count<SMUP:
        SMnum=SMLP
        while SMnum<SMUP:
            if numpy.random.random()<prob[SMnum][0]:
                count+=1
                j=numpy.random.randint(0,D)
                r1=SMnum
                while r1==SMnum:
                    r1=numpy.random.randint(SML,SMU)
                SM[SMnum][j]=SM[SMnum][j]+numpy.random.random()*(GL[j]-SM[SMnum][j])+((2*numpy.random.random())-1)*(SM[r1][j]-SM[SMnum][j])
                if SM[SMnum][j]>ub[j][0]:
                    SM[SMnum][j]=lb[j][0]+numpy.random.random()*(ub[j][0]-lb[j][0])    
                if SM[SMnum][j]<lb[j][0]:
                    SM[SMnum][j]=lb[j][0]+numpy.random.random()*(ub[j][0]-lb[j][0])               
            SMnum+=1
    SMnum=SMLP
    while SMnum<SMUP:
        SMval[SMnum][0]=func(SM[SMnum],mode=mode)
        if SMval2[SMnum][0]<SMval[SMnum][0]:
            SMval[SMnum][0]=SMval2[SMnum][0]
            SM[SMnum]=SM2[SMnum]
        else:
            SMval2[SMnum][0]=SMval[SMnum][0]
            SM2[SMnum]=SM[SMnum]
            if SMval[SMnum][0]<LLval:
                LLval=SMval[SMnum][0]
                LL=SM[SMnum]
        SMnum+=1  
    return(SM[SMLP:SMUP],SMval[SMLP:SMUP],LL,LLval,SM2[SMLP:SMUP],SMval2[SMLP:SMUP])

aa=1
SMtot=40
MSw=1
Sw=numpy.array([[1,0,SMtot,0,0,0,0,0,0,0,0,0,0,0,0,0,0],\
[0,0,SMtot/2,SMtot/2,SMtot,0,0,0,0,0,0,0,0,0,0,0,0],\
[0,0,SMtot/4,SMtot/4,SMtot/2,SMtot/2,SMtot,0,0,0,0,0,0,0,0,0,0],\
[0,0,SMtot/4,SMtot/4,SMtot/2,SMtot/2,(3*SMtot)/4,(3*SMtot)/4,SMtot,0,0,0,0,0,0,0,0],\
[0,0,SMtot/8,SMtot/8,SMtot/4,SMtot/4,SMtot/2,SMtot/2,(3*SMtot)/4,(3*SMtot)/4,SMtot,0,0,0,0,0,0],\
[0,0,SMtot/8,SMtot/8,SMtot/4,SMtot/4,(3*SMtot)/8,(3*SMtot)/8,SMtot/2,SMtot/2,(3*SMtot)/4,(3*SMtot)/4,SMtot,0,0,0,0],\
[0,0,SMtot/8,SMtot/8,SMtot/4,SMtot/4,(3*SMtot)/8,(3*SMtot)/8,SMtot/2,SMtot/2,(5*SMtot)/8,(5*SMtot)/8,(3*SMtot)/4,(3*SMtot)/4,SMtot,0,0],\
[0,0,SMtot/8,SMtot/8,SMtot/4,SMtot/4,(3*SMtot)/8,(3*SMtot)/8,SMtot/2,SMtot/2,(5*SMtot)/8,(5*SMtot)/8,(3*SMtot)/4,(3*SMtot)/4,(7*SMtot)/8,(7*SMtot)/8,SMtot]])

s0=0
sk=SMtot/8
s1=2*sk
s2=4*sk
s3=6*sk       
s4=8*sk

# boundary constraints
ub=numpy.random.random(size=(D,1))
lb=numpy.random.random(size=(D,1))
i=0
while i<D:
    ub[i][0]=8.8
    lb[i][0]=2.2
    i+=1


if aa==0:
    GLL=SMtot
    LLL=SMtot*D
elif aa==1:
    GLL=20
    LLL=500


for run_no in range(runs):
    SM=numpy.random.uniform(size=(SMtot,D))
    SM2=numpy.random.uniform(size=(SMtot,D))
    i=0
    while i<SMtot:
        j=0
        while j<D:
            SM[i][j]=lb[j][0]+SM[i][j]*(ub[j][0]-lb[j][0])
            SM2[i][j]=SM[i][j]
            j+=1
        i+=1
    SMval=numpy.random.uniform(size=(SMtot,1))
    SMval2=numpy.random.uniform(size=(SMtot,1))
    LL=SM[0:MG]
    GL=SM[1]    
    LLval=numpy.random.uniform(size=(MG,1))*1e12
    GLval=100000000
    LLc=numpy.zeros(shape=(MG,1))
    GLc=0
    
    maxerr=-10000000
    ## def optimisation variable
    ## here we have to minimise f2
    opmval=numpy.random.random(size=(6,3))
    aa1=0
    aa2=0
    aa3=0
    aa4=0
    aa5=0
    aa6=0
    
    SMnum=0
    while SMnum<SMtot:
        SMval[SMnum][0]=func(SM[SMnum],mode=mode)
        SMval2[SMnum][0]=SMval[SMnum][0]
        if SMval[SMnum][0]<LLval[0][0]:
            LLval[0][0]=SMval[SMnum][0]
            LL[0]=SM[SMnum]
        SMnum+=1
    GLval=LLval[0][0]
    GL=LL[0]
    
    prob=numpy.random.random(size=(SMtot,1))
    LLval2=numpy.random.random(size=(MG,1))
    i=0
    while i<MG:
        LLval2[i][0]=LLval[i][0]
        i+=1
        
    qq=1 
    time1=time.time()
    it=0
    fv=0
    while it<maxiter and GLval>=maxerr:
        pr=prmn+(prmx-prmn)*(it/maxiter)
        LLval2=LLval.copy()   
        if MSw==1:
    
            (SM[s0:s1],SMval[s0:s1],LL[0],LLval[0][0],SM2[s0:s1],SMval2[s0:s1])=SMO_a1(SM,SMval,Sw[0][1],Sw[0][2],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s0,s1)
            (SM[s1:s2],SMval[s1:s2],LL[0],LLval[0][0],SM2[s1:s2],SMval2[s1:s2])=SMO_a1(SM,SMval,Sw[0][1],Sw[0][2],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s1,s2)
            (SM[s2:s3],SMval[s2:s3],LL[0],LLval[0][0],SM2[s2:s3],SMval2[s2:s3])=SMO_a1(SM,SMval,Sw[0][1],Sw[0][2],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s2,s3)
            (SM[s3:s4],SMval[s3:s4],LL[0],LLval[0][0],SM2[s3:s4],SMval2[s3:s4])=SMO_a1(SM,SMval,Sw[0][1],Sw[0][2],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s3,s4)
            
                    
            (SM[s0:s1],SMval[s0:s1],LL[0],LLval[0][0],SM2[s0:s1],SMval2[s0:s1])=SMO_a2(SM,SMval,Sw[0][1],Sw[0][2],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s0,s1)
            (SM[s1:s2],SMval[s1:s2],LL[0],LLval[0][0],SM2[s1:s2],SMval2[s1:s2])=SMO_a2(SM,SMval,Sw[0][1],Sw[0][2],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s1,s2)
            (SM[s2:s3],SMval[s2:s3],LL[0],LLval[0][0],SM2[s2:s3],SMval2[s2:s3])=SMO_a2(SM,SMval,Sw[0][1],Sw[0][2],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s2,s3)
            (SM[s3:s4],SMval[s3:s4],LL[0],LLval[0][0],SM2[s3:s4],SMval2[s3:s4])=SMO_a2(SM,SMval,Sw[0][1],Sw[0][2],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s3,s4)
            
        if MSw==2:
    
            (SM[s0:s1],SMval[s0:s1],LL[0],LLval[0][0],SM2[s0:s1],SMval2[s0:s1])=SMO_a1(SM,SMval,Sw[1][3],Sw[1][4],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s0,s1)
            (SM[s1:s2],SMval[s1:s2],LL[0],LLval[0][0],SM2[s1:s2],SMval2[s1:s2])=SMO_a1(SM,SMval,Sw[1][3],Sw[1][4],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s1,s2)
            (SM[s2:s3],SMval[s2:s3],LL[0],LLval[0][0],SM2[s2:s3],SMval2[s2:s3])=SMO_a1(SM,SMval,Sw[1][3],Sw[1][4],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s2,s3)
            (SM[s3:s4],SMval[s3:s4],LL[0],LLval[0][0],SM2[s3:s4],SMval2[s3:s4])=SMO_a1(SM,SMval,Sw[1][3],Sw[1][4],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s3,s4)
                  
            (SM[s0:s1],SMval[s0:s1],LL[0],LLval[0][0],SM2[s0:s1],SMval2[s0:s1])=SMO_a2(SM,SMval,Sw[1][3],Sw[1][4],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s0,s1)
            (SM[s1:s2],SMval[s1:s2],LL[0],LLval[0][0],SM2[s1:s2],SMval2[s1:s2])=SMO_a2(SM,SMval,Sw[1][3],Sw[1][4],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s1,s2)
            (SM[s2:s3],SMval[s2:s3],LL[0],LLval[0][0],SM2[s2:s3],SMval2[s2:s3])=SMO_a2(SM,SMval,Sw[1][3],Sw[1][4],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s2,s3)
            (SM[s3:s4],SMval[s3:s4],LL[0],LLval[0][0],SM2[s3:s4],SMval2[s3:s4])=SMO_a2(SM,SMval,Sw[1][3],Sw[1][4],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s3,s4)
            
        if MSw==3:
    
            (SM[s0:s1],SMval[s0:s1],LL[0],LLval[0][0],SM2[s0:s1],SMval2[s0:s1])=SMO_a1(SM,SMval,Sw[2][5],Sw[2][6],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s0,s1)
            (SM[s1:s2],SMval[s1:s2],LL[0],LLval[0][0],SM2[s1:s2],SMval2[s1:s2])=SMO_a1(SM,SMval,Sw[2][5],Sw[2][6],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s1,s2)
            (SM[s2:s3],SMval[s2:s3],LL[1],LLval[1][0],SM2[s2:s3],SMval2[s2:s3])=SMO_a1(SM,SMval,Sw[2][5],Sw[2][6],D,LL[1],LLval[1][0],pr,SM2,SMval2,LLval2[1][0],LLL,prob,ub,lb,GL,s2,s3)
            (SM[s3:s4],SMval[s3:s4],LL[1],LLval[1][0],SM2[s3:s4],SMval2[s3:s4])=SMO_a1(SM,SMval,Sw[2][5],Sw[2][6],D,LL[1],LLval[1][0],pr,SM2,SMval2,LLval2[1][0],LLL,prob,ub,lb,GL,s3,s4)
                   
            (SM[s0:s1],SMval[s0:s1],LL[0],LLval[0][0],SM2[s0:s1],SMval2[s0:s1])=SMO_a2(SM,SMval,Sw[2][5],Sw[2][6],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s0,s1)
            (SM[s1:s2],SMval[s1:s2],LL[0],LLval[0][0],SM2[s1:s2],SMval2[s1:s2])=SMO_a2(SM,SMval,Sw[2][5],Sw[2][6],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s1,s2)
            (SM[s2:s3],SMval[s2:s3],LL[1],LLval[1][0],SM2[s2:s3],SMval2[s2:s3])=SMO_a2(SM,SMval,Sw[2][5],Sw[2][6],D,LL[1],LLval[1][0],pr,SM2,SMval2,LLval2[1][0],LLL,prob,ub,lb,GL,s2,s3)
            (SM[s3:s4],SMval[s3:s4],LL[1],LLval[1][0],SM2[s3:s4],SMval2[s3:s4])=SMO_a2(SM,SMval,Sw[2][5],Sw[2][6],D,LL[1],LLval[1][0],pr,SM2,SMval2,LLval2[1][0],LLL,prob,ub,lb,GL,s3,s4)
               
        
        if MSw==4:
    
            (SM[s0:s1],SMval[s0:s1],LL[0],LLval[0][0],SM2[s0:s1],SMval2[s0:s1])=SMO_a1(SM,SMval,Sw[3][7],Sw[3][8],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s0,s1)
            (SM[s1:s2],SMval[s1:s2],LL[0],LLval[0][0],SM2[s1:s2],SMval2[s1:s2])=SMO_a1(SM,SMval,Sw[3][7],Sw[3][8],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s1,s2)
            (SM[s2:s3],SMval[s2:s3],LL[1],LLval[1][0],SM2[s2:s3],SMval2[s2:s3])=SMO_a1(SM,SMval,Sw[3][7],Sw[3][8],D,LL[1],LLval[1][0],pr,SM2,SMval2,LLval2[1][0],LLL,prob,ub,lb,GL,s2,s3)
            (SM[s3:s4],SMval[s3:s4],LL[1],LLval[1][0],SM2[s3:s4],SMval2[s3:s4])=SMO_a1(SM,SMval,Sw[3][7],Sw[3][8],D,LL[1],LLval[1][0],pr,SM2,SMval2,LLval2[1][0],LLL,prob,ub,lb,GL,s3,s4)
                 
            (SM[s0:s1],SMval[s0:s1],LL[0],LLval[0][0],SM2[s0:s1],SMval2[s0:s1])=SMO_a2(SM,SMval,Sw[3][7],Sw[3][8],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s0,s1)
            (SM[s1:s2],SMval[s1:s2],LL[0],LLval[0][0],SM2[s1:s2],SMval2[s1:s2])=SMO_a2(SM,SMval,Sw[3][7],Sw[3][8],D,LL[0],LLval[0][0],pr,SM2,SMval2,LLval2[0][0],LLL,prob,ub,lb,GL,s1,s2)
            (SM[s2:s3],SMval[s2:s3],LL[1],LLval[1][0],SM2[s2:s3],SMval2[s2:s3])=SMO_a2(SM,SMval,Sw[3][7],Sw[3][8],D,LL[1],LLval[1][0],pr,SM2,SMval2,LLval2[1][0],LLL,prob,ub,lb,GL,s2,s3)
            (SM[s3:s4],SMval[s3:s4],LL[1],LLval[1][0],SM2[s3:s4],SMval2[s3:s4])=SMO_a2(SM,SMval,Sw[3][7],Sw[3][8],D,LL[1],LLval[1][0],pr,SM2,SMval2,LLval2[1][0],LLL,prob,ub,lb,GL,s3,s4)    
                    
           
        i1=0
        while i1<MSw:
            if LLval[i1][0]==LLval2[i1][0]:
                LLc[i1][0]+=1
            else:
                LLc[i1][0]=0      
            i1+=1
    
        i=0
        GLval1=GLval
        while i<MSw:
            if LLval[i][0]<GLval:
                GLval=LLval[i][0]
                GL=LL[i]
            i+=1
        if GLval==GLval1:
            GLc+=1
        else:
            GLc=0
        grp=1
        while grp<=MSw:
            if LLc[grp-1][0]>=LLL:
                LLc[grp-1][0]=0
                SMnum=Sw[MSw-1][2*grp-1]
                while SMnum<Sw[MSw-1][2*grp]:
                    j=0
                    while j<D:
                        if numpy.random.random()>pr:
                            SM[SMnum][j]=lb[j][0]+numpy.random.random()*(ub[j][0]-lb[j][0])
                        else:
                            SM[SMnum][j]=SM[SMnum][j]+numpy.random.random()*(GL[j]-SM[SMnum][j])-numpy.random.random()*(LL[grp-1][j]-SM[SMnum][j])
                        if SM[SMnum][j]>ub[j][0]:
                            #SM[SMnum][j]=SM[SMnum][j]-1.1*(SM[SMnum][j]-ub[j][0])
                            SM[SMnum][j]=lb[j][0]+numpy.random.random()*(ub[j][0]-lb[j][0])    
                        if SM[SMnum][j]<lb[j][0]:
                            SM[SMnum][j]=lb[j][0]+numpy.random.random()*(ub[j][0]-lb[j][0])    
                        j+=1
                    SMnum+=1
            grp+=1
        if GLc>=GLL:
            GLc=0
            if MSw<MG:
                MSw+=1
                if MSw==2:
                    LLc[0][0]=0
                    LLc[1][0]=0
                elif MSw==3:
                    LLc[2][0]=LLc[1][0]
                    LLc[0][0]=0
                    LLc[1][0]=0
                elif MSw==4:
                    LLc[2][0]=0
                    LLc[3][0]=0 
                elif MSw==5:
                    LLc[4][0]=LLc[3][0]
                    LLc[3][0]=LLc[2][0]
                    LLc[2][0]=LLc[1][0]
                    LLc[0][0]=0
                    LLc[1][0]=0
                elif MSw==6:
                    LLc[5][0]=LLc[4][0]
                    LLc[4][0]=LLc[3][0]
                    LLc[2][0]=0
                    LLc[3][0]=0
                elif MSw==7:
                    LLc[6][0]=LLc[5][0]
                    LLc[4][0]=0
                    LLc[5][0]=0
                elif MSw==8:
                    LLc[6][0]=0
                    LLc[7][0]=0
            else:
                MSw=1
                grp=1
                while grp<=MG:
                    LLc[grp-1][0]=0
                    grp+=1                           
            grp=1
            print 'MSw',MSw
            while grp<=MSw:
                SMnum=Sw[MSw-1][2*grp-1]
                LLval[grp-1][0]=SMval[SMnum][0]
                LL[grp-1]=SM[SMnum]
                SMnum+=1
                while SMnum<Sw[MSw-1][2*grp]:
                    if SMval[SMnum][0]<LLval[grp-1][0]:
                        LLval[grp-1][0]=SMval[SMnum][0]
                        LL[grp-1]=SM[SMnum]
                    SMnum+=1
                grp+=1
        print 'it=',it,' MSw=',MSw,' GLval=',GLval

        fitness_val[run_no][it]=GLval              
        it+=1
    best_pos[run_no]=GL.copy()
    time2=time.time()
    print time2-time1
    run_no+=1
numpy.savetxt("DE_fitness_d1_m2"+str(mode)+str(D)+".csv",fitness_val,delimiter=",")
numpy.savetxt("DE_bestpos_d1_m2"+str(mode)+str(D)+".csv",best_pos,delimiter=",")

