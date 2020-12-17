from __future__ import division
import os, glob
import errno
import sys
import subprocess
import numpy as np
import time
import pandas as pd
import re
sys.path.append('/home/psr/software/sigpyproc')
import sigpyproc
from sigpyproc.Readers import readTim, readDat, FilReader
from joblib import Parallel, delayed
from sigpyproc.Readers import readDat, readTim, FilReader


def correlation(x1,x2):
	a1=x1[200:5499800]
	c=np.correlate(x1,x2,mode='valid')
	g=c.argmax()
	a2=x2[200:5499800]
	c1=np.correlate(x2,x1,mode='valid')	
	g1=c1.argmax()
	k=0
	m=np.absolute(g-200)
	if m==0:
		#g=g1
		#no delay
		k=0
	else:
		if g>g1:
		#x1 leads x2
			k=1
		else: #g>g1:
			k=2
	return m,k

def dedisperse(a,b):
	f=filterbank+'beam%s/'%(b)+'*fil'
	data=FilReader(f)
	z=data.dedisperse(0)
	return z
	
def splitFil(filterbank, l2):
	for beam in ['00','01','02','03','04','05','06']:
		f1=filterbank+'beam%s/'%(beam)+'*fil'
		data=FilReader(f1)
		z1=data.split(0,l2, filename=f1)
		
def prepfil(filterbank, b):
	dat=FilReader(filterbank+'beam%s/'%(b)+'*.fil'
	fil=dat.readPlan(10000, skipback=0, start=0, nsamps=None, verbose=True)
	filar=[]
	for nsamps, ii, data in fil:
		data = data.reshape(nsamps,dat.header.nchans)
		filar.append(data)

	w=np.concatenate(filar,axis=0)
	return w

def delay_calc(b5,b6,e2,sig1,sig2,ar1,ar2,q4,j4,qindex): #b5 is first beam, b6 is second, ar1 and ar2 are the timeseries for the current frequency channel for the 2 beams
	ar=0
	if q4[qindex]==1:
		for k6 in range(0, len(e2),1):
			if e2[k6][2]==b5 or e2[k6][3]==b5:
				if e2[k6][3]==b6 or e2[k6][2]==b6:
					if e2[k6][0]>60:
						ar2s=np.avg(ar2[(j4-e2[k6][0])-1],ar2[j4-e2[k6][0]],ar2[(j4-e2[k6][0])+1])
						if ar2s>(1.5*sig2):
							ar=1
					else:
						e2[k6][0]=0
						ar2s=np.avg(ar2[(j4-e2[k6][0])-1],ar2[j4-e2[k6][0]],ar2[(j4-e2[k6][0])+1])
                                                if ar2s>(1.5*sig2):
                                                        ar=1
	else:
		for k6 in range(0, len(e2),1):
			if e2[k6][3]==b5 and e2[k6][2]==b5:
				if e2[k6][3]==b6 or e2[k6][2]==b6:
					if e2[k6][0]>60:
						arss2=np.avg(ar2[(j4+e2[k6][0])-1],ar2[j4+e2[k6][0]],ar2[(j4+e2[k6][0])+1])
						if arss2>(1.5*sig2):
							ar=1             
					else:
						e2[k6][0]=0
						arss2=np.avg(ar2[(j4+e2[k6][0])-1],ar2[j4+e2[k6][0]],ar2[(j4+e2[k6][0])+1])
                                                if arss2>(1.5*sig2):
                                                        ar=1


	return ar

def compare_samps(o1,e1,p0,p1,p2,p3,p4,p5,p6, pp0, pp1, pp2, pp3, pp4, pp5, pp6, j3, sg0, sg1, sg2, sg3, sg4, sg5, sg6):
	m3=o1[0]                            #e1 is the delay arrays containing the two combos of the 4 beams sent, for eg id 4 beams are 1234, e has 12,13,14,23,24,24.
	q3=[]
	for n in range(1,4,1):              #to decide if the rest of the beams apart from the first are leading or lagging wrt to first
		m5=o1[n]
		for y in range(0,len(e1),1):
			if e1[y][2]==m3 or e1[y][2]==m5:
				if e1[y][3]==m3 or e1[y][3]==m5:
					if m3==e1[y][2]:
						if e1[y][1]==1:
							q3.append(1)
						else:
							q3.append(0)
 #1 is m3 or first ebam of the combo si ahead wrt to the current beam ahead in time 0, that is m3 is leading the other array elements is lagging
					else:
						if e1[y][1]==1:
							q3.append(0)
						else:
							q3.append(1)
#q3 is array of 0 and/or 1s, which says the remaining 3 beams are leading or lagging wrt 1 st beam, if 1, 1st beam is leading, 0, nth beam is leading the 1s t beam

	r0=0
	r1=0
	r2=0
	r3=0
	r4=0
	r5=0
	r6=0

	if 0 in o1: #checking which beams are in the current combo
		r0=1
	if 1 in o1:
		r1=1
	if 2 in o1:
		r2=1
	if 3 in o1:
		r3=1
	if 4 in o1:
		r4=1
	if 5 in o1:
		r5=1
	if 6 in o1:
		r6=1


        u0=0
	u1=0
	u2=0
	u3=0
	u4=0
	u5=0
	u6=0

	if r0==1:		#checking which beam is the first in the combo, needed for sign of the delays
		if 0==m3:
			u0=1

	if r1==1:
                if 1==m3:
                        u1=1

	if r2==1:
                if 2==m3:
                        u2=1
	
	if r3==1:
                if 3==m3:
                        u3=1

	if r4==1:
                if 4==m3:
                        u4=1

	if r5==1:
                if 5==m3:
                        u5=1
	
	if r6==1:
                if 6==m3:
                        u6=1

	
	array=[u0,u1,u2,u3,u4,u5,u6]
#	for h5 in range(0,7,1):
	a0=a1=a2=a3=a4=a5=a6=0
	if u0==1:
		if p0>(1.5*sg0):
			a0=1
		for qn in range(1,4,1):
			a1=fed_up(0,1,o1,e1,sg0,sg1,pp0,pp1,q3,j3,qn)
			a2=fed_up(0,2,o1,e1,sg0,sg2,pp0,pp2,q3,j3,qn)
			a3=fed_up(0,3,o1,e1,sg0,sg3,pp0,pp3,q3,j3,qn)
			a4=fed_up(0,4,o1,e1,sg0,sg4,pp0,pp4,q3,j3,qn)
			a5=fed_up(0,5,o1,e1,sg0,sg5,pp0,pp5,q3,j3,qn)
			a6=fed_up(0,6,o1,e1,sg0,sg6,pp0,pp6,q3,j3,qn)

	if u1==1:
		if pp1[j3]>1.5*sg1:
			a1=1
		for qn in range(1,4,1):
                        a0=fed_up(1,0,o1,e1,sg1,sg0,pp1,pp0,q3,j3,qn)
                        a2=fed_up(1,2,o1,e1,sg1,sg2,pp1,pp2,q3,j3,qn)
                        a3=fed_up(1,3,o1,e1,sg1,sg3,pp1,pp3,q3,j3,qn)
                        a4=fed_up(1,4,o1,e1,sg1,sg4,pp1,pp4,q3,j3,qn)
                        a5=fed_up(1,5,o1,e1,sg1,sg5,pp1,pp5,q3,j3,qn)
                        a6=fed_up(1,6,o1,e1,sg1,sg6,pp1,pp6,q3,j3,qn)


	if u2==1:
		if pp2[j3]>1.5*sg2:
			a2=1
		for qn in range(1,4,1):
                        a0=fed_up(2,0,o1,e1,sg2,sg0,pp2,pp0,q3,j3,qn)
                        a1=fed_up(2,1,o1,e1,sg2,sg1,pp2,pp1,q3,j3,qn)
                        a3=fed_up(2,3,o1,e1,sg2,sg3,pp2,pp3,q3,j3,qn)
                        a4=fed_up(2,4,o1,e1,sg2,sg4,pp2,pp4,q3,j3,qn)
                        a5=fed_up(2,5,o1,e1,sg2,sg5,pp2,pp5,q3,j3,qn)
                        a6=fed_up(2,6,o1,e1,sg2,sg6,pp2,pp6,q3,j3,qn)

	if u3==1:
		if pp3[j3]>1.5*sg3:
			a3=1
                for qn in range(1,4,1):
                        a0=fed_up(3,0,o1,e1,sg3,sg0,pp3,pp0,q3,j3,qn)
                        a1=fed_up(3,1,o1,e1,sg3,sg1,pp3,pp1,q3,j3,qn)
                        a2=fed_up(3,2,o1,e1,sg3,sg2,pp3,pp2,q3,j3,qn)
                        a4=fed_up(3,4,o1,e1,sg3,sg4,pp3,pp4,q3,j3,qn)
                        a5=fed_up(3,5,o1,e1,sg3,sg5,pp3,pp5,q3,j3,qn)
                        a6=fed_up(3,6,o1,e1,sg3,sg6,pp3,pp6,q3,j3,qn)

	if u4==1:
		if pp4[j3]>1.5*sg4;
			a4=1
		for qn in range(1,4,1):
                        a0=fed_up(4,0,o1,e1,sg4,sg0,pp4,pp0,q3,j3,qn)
                        a1=fed_up(4,1,o1,e1,sg4,sg1,pp4,pp1,q3,j3,qn)
                        a2=fed_up(4,2,o1,e1,sg4,sg2,pp4,pp2,q3,j3,qn)
                        a3=fed_up(4,3,o1,e1,sg4,sg3,pp4,pp3,q3,j3,qn)
                        a5=fed_up(4,5,o1,e1,sg4,sg5,pp4,pp5,q3,j3,qn)
                        a6=fed_up(4,6,o1,e1,sg4,sg6,pp4,pp6,q3,j3,qn)

	if u5==1:
		if pp5[j3]>1.5*sg5:
			a5=1
		for qn in range(1,4,1):
                        a0=fed_up(5,0,o1,e1,sg5,sg0,pp5,pp0,q3,j3,qn)
                        a1=fed_up(5,1,o1,e1,sg5,sg1,pp5,pp1,q3,j3,qn)
                        a2=fed_up(5,2,o1,e1,sg5,sg2,pp5,pp2,q3,j3,qn)
                        a4=fed_up(5,4,o1,e1,sg5,sg4,pp5,pp4,q3,j3,qn)
                        a3=fed_up(5,3,o1,e1,sg5,sg3,pp5,pp3,q3,j3,qn)
                        a6=fed_up(5,6,o1,e1,sg5,sg6,pp5,pp6,q3,j3,qn)

	if u6==1:
		if pp6[j3]>1.5*sg6:
			a6=1
		for qn in range(1,4,1):
                        a0=fed_up(6,0,o1,e1,sg6,sg0,pp6,pp0,q3,j3,qn)
                        a1=fed_up(6,1,o1,e1,sg6,sg1,pp6,pp1,q3,j3,qn)
                        a2=fed_up(6,2,o1,e1,sg6,sg2,pp6,pp2,q3,j3,qn)
                        a4=fed_up(6,4,o1,e1,sg6,sg4,pp6,pp4,q3,j3,qn)
                        a5=fed_up(6,5,o1,e1,sg6,sg5,pp6,pp5,q3,j3,qn)
                        a3=fed_up(6,3,o1,e1,sg6,sg3,pp6,pp3,q3,j3,qn)


	return a0,a1,a2,a3,a4,a5,a6
			
def fed_up(hell1,hell2,o2,e12,sgg0,sgg1,ppp0,ppp1,q33,j33,qin):
	if o2[qin]==hell2:
		azz=delay_calc(hell1,hell2,e12,sgg0,sgg1,ppp0,ppp1,q33,j33,qin)
	else:
		azz=0
	return azz


def append_bins(r440,r441,oin,e5,f11,f12,t11,t12): #function to append appropriate samples with dleays from analysis

        for h9 in range(len(e5)):
                if e5[h9][2]==r440 or e5[h9][3]==r440:
                        if e5[h9][3]==r441 or e5[h9][2]==r441:
                                if e5[h9][2]==0:
                                        if e5[h9][1]==1:
                                                if e5[h9][0]<60:                                                        #0 leads 1
                                                        f11.append(i)
                                                        f12.append(i)
                                                        t11.append(j)
                                                        t12.append(j)
                                                else:
                                                        t11.append(j)
                                                        t12.append(j-e[h9][0])
                                                        f11.append(i)
                                                        f12.append(i)
                                        else:#1 leads 0
                                                if e5[h9][0]<60:                                                        #0 leads 1
                                                        f11.append(i)
                                                        f12.append(i)
                                                        t11.append(j)
                                                        t12.append(j)
                                                else:
                                                        t11.append(j)
                                                        t12.append(j+e[h9][0])
                                                        f11.append(i)
                                                        f12.append(i)



#final stage
#writing the new filterbanks with modified values

def write_fil(finals,f8,t8,filterbank,beam,kl):
        sum1=0
        for x11 in range(0,512,1):
                for y11 in range(0,kl,1):
                        for z11 in range(0,len(f8),1):
                                if y11==t8[z11] and x11=f8[z11]:
                                        for z12 in range(-5000,5000,1):
                                                sum1=sum1+finals[y11+z12][x11]
                                        avg=sum1/10000

                                        finals[y11][x11]=avg


        dat=FilReader(filterbank+'/beam0%s/*fil'%(beam))
        s11=dat.header.prepOutfile(filterbank+'/beam0%s/new.fil'%(beam),nbits=8)
        s.cwrite(finals.ravel())
        s.close()

























if __name__ == '__main__':
		
	pointing=sys.argv[1]
	filterbank='/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/filterbank_files/'
	beam=['00','01','02','03','04','05','06']
	d0=dedisperse(filterbank, '00')
	d1=dedisperse(filterbank, '01')
	d2=dedisperse(filterbank, '02')
	d3=dedisperse(filterbank, '03')
	d4=dedisperse(filterbank, '04')
	d5=dedisperse(filterbank, '05')
	d6=dedisperse(filterbank, '06')
	

	z0=d0[0:5500000]
	z1=d1[0:5500000]
	z2=d2[0:5500000]
	z3=d3[0:5500000]
	z4=d4[0:5500000]
	z5=d5[0:5500000]
	z6=d6[0:5500000]

#	val = Parallel(n_jobs = 21)(delayed(correlation)(current_filterbank, number_samples8, mask_file, dm) for dm in np.arange(1808.136, 3002, 7.063))
	t01, k01= correlation(z0,z1)
	t02, k02= correlation(z0,z2)
	t03, k03= correlation(z0,z3)
	t04, k04= correlation(z0,z4)
	t05, k06= correlation(z0,z5)
	t06, k06= correlation(z0,z6)
	t12, k12= correlation(z1,z2)
	t13, k13= correlation(z1,z3)
	t14, k14= correlation(z1,z4)
	t15, k15= correlation(z1,z5)
	t16, k16= correlation(z1,z6)
	t23, k23= correlation(z2,z3)
	t24, k24= correlation(z2,z4)
	t25, k25= correlation(z2,z5)
	t26, k26= correlation(z2,z6)
	t34, k34= correlation(z3,z4)
	t35, k35= correlation(z3,z5)
	t36, k36= correlation(z3,z6)
	t45, k45= correlation(z4,z5)
	t46, k46= correlation(z4,z6)
	t56, k56= correlation(z5,z6)

	c=[[t01,k01,0,1],[t02,k02,0,2],[t03,k03,0,3],[t04,k04,0,4],[t05,k05,0,5],[t06,k06,0,6],[t12,k12,1,2],[t13,k13,1,3],[t14,k14,1,4],[t15,k15,1,5],[t16,k16,1,6],[t23,k23,2,3],[t24,k24,2,4],[t25,k25,2,5],[t26,k26,2,6],[t34,t34,3,4],[t35,k35,3,5],[t36,k36,3,6],[t45,k45,4,5],[t46,k46,4,6],[t56,k56,5,6]]

#if k=1, first array zx is leading the second, m is the number of samples by which. k=2 swxond is leading the first. we go for delay correction only if m > 60 samples, which is the expected delay between the pffts, else straightaway to comaprisons.
        
	a=np.array([len(d0), len(d1), len(d2), len(d3), len(d4), len(d5), len(d6)]) 
	l=min(a)
	l1=a.argmin()

	h=[0,1,2,3,4,5,6]
	b=list(itertools.combinations(h,4))
	
	splitFil(filterbank,l1)
	#making the filterbanks all of the same size

	

	final0=prep_fil(filterbank,'00')
	final1=prep_fil(filterbank,'01')
	final2=prep_fil(filterbank,'02')
	final3=prep_fil(filterbank,'03')
	final4=prep_fil(filterbank,'04')
	final5=prep_fil(filterbank,'05')
	final6=prep_fil(filterbank,'06')

	t0=[]
	t1=[]
	t2=[]
	t3=[]
	t4=[]
	t5=[]
	t6=[]
	f0=[]
	f1=[]
	f2=[]
	f3=[]
	f4=[]
	f5=[]
	f6=[]
	
	hel=max(t01,t02,t03,t04,t05,t06,t12,t13,t14,t15,t16,t23,t24,t25,t26,t34,t35,t36,t45,t46,t56)
	
	for i in range(0,512,1):
		for j in range(hel,l1-hel,1):
		#	for k in range(0,j,1):
			c0=0
			c1=0
			c2=0
			c3=0
			c4=0
			c5=0
			c6=0
			sd0=np.std(final0[:,j])
			sd1=np.std(final0[:,j])
			sd2=np.std(final2[:,j])
			sd3=np.std(final3[:,j])
			sd4=np.std(final4[:,j])
			sd5=np.std(final5[:,j])
			sd6=np.std(final6[:,j])

			for k in range(0,len(b),1):
				#looping over 4 beam combinations
				o=b[k] #the 4 beams considering now
				v=itertools.combinations(b[k],2)
				e=[]
				for z in range(0,6,1):
					x4=v[z]
					for z1 in range(0,21,1):
						q=c[z1]
						if c[z1][2] in x4 and c[z1][3] in x4:
							e.append(c[z1])	#the array containing the mutual delays of the current combination of 4 beams
					#sending j th time samp for ith frequency channel for comparisons

				fj0=np.avg(final0[j-1][i],final0[j][i],final0[j+1][i])
				fj1=np.avg(final1[j-1][i],final1[j][i],final1[j+1][i])
				fj2=np.avg(final2[j-1][i],final2[j][i],final2[j+1][i])
				fj3=np.avg(final3[j-1][i],final3[j][i],final3[j+1][i])
				fj4=np.avg(final4[j-1][i],final4[j][i],final4[j+1][i])
				fj5=np.avg(final5[j-1][i],final5[j][i],final5[j+1][i])
				fj6=np.avg(final6[j-1][i],final6[j][i],final6[j+1][i])

				c0,c1,c2,c3,c4,c5,c6=compare_samps(o, e, fj0, fj1,fj2,fj3,fj4,fj5,fj6, final0[:,j], final1[:,j], final2[:,j],final3[:,j], final4[:,j], final5[:,j], final6[:,j],j,sd0,sd1,sd2,sd3,sd4,sd5,sd6) 	



				counter=[c0,c1,c2,c3,c4,c5,c6]
				b11=list(itertools.combinations(counter,4))
				for h8 in range(0, len(b11), 1):
					if b11[h8][0]==1 and b11[h8][1]==1 and b11[h8][2]==1 and b11[h8][3]==1:
						#o=b[k] current combo has bad bins in 4 beams
						c40=c41=c42=c43=c45=c46=c44=0
						if 0 in o:	
							c40=1
						if 1 in o:
							c41=1
						if 2 in o:
							c42=1
						if 3 in o:
							c43=1
						if 4 in o:
							c44=1
						if 5 in o:
							c45=1
						if 6 in o:
							c46=1

						r40=r41=r42=r43=r45=r44=r46=0

						if c40==1:
							if o[0]==0:
								r40=1
						if c41==1:
                                                        if o[0]==1:
								r41=1
                                                if c42==1:
                                                        if o[0]==2:
                                                                r42=1 

						if c43==1:
							if o[0]==3:
								r43=1
						if c44==1:
							if o[0]==4:
								r44=1
						if c45==1:
							if o[0]==5:
								r45=1

						if c46==1:
							if o[0]==6:
								r46=1


						if r40==1:
							for k7 in range(1,4,1):
								if o[k7]==1:
									append_bins(0,1,k7,e,f0,f1,t0,t1)

								if o[k7]==2:
									append_bins(0,2,k7,e,f0,f2,t0,t2)
		
								if o[k7]==3:
									append_bins(0,3,k7,e,f0,f3,t0,t3)
								if o[k7]==4:
									append_bins(0,4,k7,e,f0,f4,t0,t4)
								if o[k7]==5:
									append_bins(0,5,k7,e,f0,f5,t0,t5)
								if o[k7]==6:
									append_bins(0,6,k7,e,f0,f6,t0,t6)



						if r41==1:
							for k7 in range(1,4,1):
                                                                if o[k7]==0:
                                                                        append_bins(1,0,k7,e,f1,f0,t1,t0)

                                                                if o[k7]==2:
                                                                        append_bins(1,2,k7,e,f1,f2,t1,t2)

                                                                if o[k7]==3:
                                                                        append_bins(1,3,k7,e,f1,f3,t1,t3)
                                                                if o[k7]==4:
                                                                        append_bins(1,4,k7,e,f1,f4,t1,t4)
                                                                if o[k7]==5:
                                                                        append_bins(1,5,k7,e,f1,f5,t1,t5)
                                                                if o[k7]==6:
                                                                        append_bins(1,6,k7,e,f1,f6,t1,t6)

                                              #repeat for other ebams if r20, r30, r40, r50, r60==1...

#final step outputting the modified filterbank files


	write_fil(final0,f0,t0,filterbank,0,l1)
	write_fil(final1,f1,t1,filterbank,1,l1)
	write_fil(final2,f2,t2,filterbank,2,l1)
	write_fil(final3,f3,t3,filterbank,3,l1)
	write_fil(final4,f4,t4,filterbank,4,l1)
	write_fil(final5,f5,t5,filterbank,5,l1)
	write_fil(final6,f6,t6,filterbank,6,l1)



'''def append_bins(r440,r441,oin,e5,f11,f12,t11,t12): #function to append appropriate samples with dleays from analysis

	for h9 in range(len(e5)):
		if e5[h9][2]==r440 or e5[h9][3]==r440:
			if e5[h9][3]==r441 or e5[h9][2]==r441:
				if e5[h9][2]==0:
					if e5[h9][1]==1:
                                                if e5[h9][0]<60:                                                        #0 leads 1
							f11.append(i)
							f12.append(i)
							t11.append(j)
							t12.append(j)
						else:
							t11.append(j)
                                                        t12.append(j-e[h9][0])
                                                        f11.append(i)
                                                        f12.append(i)
					else:#1 leads 0
						if e5[h9][0]<60:                                                        #0 leads 1
                                                        f11.append(i)
                                                        f12.append(i)
                                                        t11.append(j)
                                                        t12.append(j)
						else:
                                                	t11.append(j)
                                                	t12.append(j+e[h9][0])
                                                	f11.append(i)
                                                	f12.append(i)
						


#final stage
#writing the new filterbanks with modified values

def write_fil(finals,f8,t8,filterbank,beam,kl):
	sum1=0
	for x11 in range(0,512,1):
		for y11 in range(0,kl,1):
			for z11 in range(0,len(f8),1):
				if y11==t8[z11] and x11=f8[z11]:
					for z12 in range(-5000,5000,1):
						sum1=sum1+finals[y11+z12][x11]
					avg=sum1/10000
	
	            			finals[y11][x11]=avg


	dat=FilReader(filterbank+'/beam0%s/*fil'%(beam))
	s11=dat.header.prepOutfile(filterbank+'/beam0%s/new.fil'%(beam),nbits=8)
	s.cwrite(finals.ravel())
	s.close()'''




