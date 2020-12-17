import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import itertools
from itertools import combinations
#from sigpyproc import Utils, FilReader
from scipy.fftpack import rfft, irfft, rfftfreq
from scipy.optimize import curve_fit
#from __future__ import division
import os, glob
import errno
import sys
import subprocess
#import numpy as np
import time
import pandas as pd
import re
sys.path.append('/home/psr/software/sigpyproc')
import sigpyproc
from sigpyproc.Readers import readTim, readDat, FilReader
from joblib import Parallel, delayed
from sigpyproc.Readers import readDat, readTim, FilReader

def dered_norm_fp(beam, pointing):

	#data=FilReader('3794_0001_00_8bit.fil')
	filterbank='/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/filterbank_files/' + 'beam' +beam + '/'+ pointing + '_%s_' %str(beam) + '8bit.fil'
	data=FilReader(filterbank)
	des=data.dedisperse(0)
	#des=[]
	l=len(des)-33554432
	des=des[:-l]
	time=[]
	for i in range(0, 33554432, 1):
		j=i*5.46133333333333e-05
		time.append(j)


	fourier_trans=np.fft.rfft(des)
	fourier_amp=np.absolute(fourier_trans)
	fourier_pow=np.square(fourier_amp)
	t=time[1]-time[0]
	fourier_freq_res=np.fft.rfftfreq(des.size, d=t)
	fourier_amp=fourier_amp

#red noise removal chunk
	sum=0
	red_med=np.median(fourier_amp[0:8060])
 #running rms of the red noise region

	full=[]

# steep red noise window till about 0.13Hz
	for i in range(0, 210, 1):

        	j=i+3
        	sum=0
        	run_med=np.median(fourier_amp[i:j])
      #  for s in range(i, j, 1):
       		k=(fourier_amp[i]/run_med)*0.8325546
       		full.append(k)
              #  sum=sum+(k**2)
 

	for i in range(210, 8060, 1):

        	j=i+10
        	sum=0
        	run_med=np.median(fourier_amp[i:j])
      #  for s in range(i, j, 1):
		k=(fourier_amp[i]/run_med)*0.8325546
		full.append(k)
              #  sum=sum+(k**2)
   
 
	print ('length of array till now ', len(full))
	fa_l=len(fourier_amp)
	factor= (fa_l-8060)/10000
	fac=factor-1
	g1=fac*10000
	print ('value of fac is', fac)

	print ('value of g is', g1)
#running median and for the rest and normalising later

	med=np.mean(fourier_amp[4830:fa_l])


#running median and rms normalisation for the rest of the fourier series
	g=g1+8060
	for i in range(8060, g, 10000):
#	print (i)
		j=i+10000
		sum=0
		run_med=np.median(fourier_amp[i:j])
		for s in range(i, j, 1):
			k=(fourier_amp[s]/run_med)*0.8325546
			full.append(k)		


	print ('last iteration value of i is ',i)
	print ('length of array till now ', len(full))
	m=fa_l-g	
	print ('this value should be > 10000', m)
	run_med_last=np.median(fourier_amp[g:fa_l])
	sum1=0
	for i in range(g, fa_l, 1):
		k = (fourier_amp[i]/run_med_last)*0.8325546
		full.append(k)


	print ('mean of the full array is ', np.mean(full))
	print ('standard deviation of the full array is ', np.std(full))	
	sum_f=0
	l1=len(full)
	sd=np.std(full)

	print ('std 2:200', np.std(full[2:200]))
	print ('std 2000:5000', np.std(full[2000:5000]))
	print ('std 8500:13000', np.std(full[8500:13000]))
#plt.plot(fourier_freq_res[9000:1000000], fourier_pow[9000:1000000])
#	plt.grid(True)

	x1=[]
	x3=[]
	for i in np.arange(0, 13, 0.0001):
		j=-1*i
		k=j/2
		x1.append(j)
		x3.append(k)

	print (len(full))
	print (len(fourier_freq_res))
	print (len(fourier_amp))
#print (len(power))
	lf=len(full)


	print ('mean of the full array is ', np.mean(full))
	print ('standard deviation of the full array is ', np.std(full))

	full1=np.square(full)
#	plt.plot(fourier_freq_res[100:30000], full1[100:30000])


	print (fourier_freq_res)
	print (np.mean(full))
	x=[]

	m=[]
	for i in range(0, lf, 1):
		k=full1[i]
		if k<15625:
			m.append(k)
	#g, bins, p= plt.hist(full1[1:], 100000, edgecolor='black', color='white', normed=True, label='Normalised fourier power histogram')
	#g, bins, p= plt.hist(fourier_pow[100:], 100000, edgecolor='black', color='white', normed=True)


	a=[]
	b=[]
	for x in np.arange(0, 1, 0.0001):
		y=(-0.32*x) + 0.67
		a.append(x)
		b.append(y)

	for x in np.arange(1, 2, 0.0001):
        	y=(-0.17*x) + 0.52
        	a.append(x)
        	b.append(y)

	for x in np.arange(2, 3, 0.0001):
        	y=(-0.09*x) + 0.36
        	a.append(x)
       	        b.append(y)

	for x in np.arange(3, 4, 0.0001):
        	y=(-0.0405*x) + 0.2115
        	a.append(x)
        	b.append(y)

	for x in np.arange(4, 5.5, 0.0001):
        	y=(-0.019667*x) + 0.128167
        	a.append(x)
        	b.append(y)

#plt.plot(a, b, color='yellow')


	print ('median fp 100:400', np.median(fourier_pow[100:400]))
	print ('median fp 10000:20000', np.median(fourier_pow[10000:20000]))





#g, bins, p= plt.hist(m, 50, edgecolor='black', color='white', normed=True)
	x2=[]
	for i in np.arange(0, 13, 0.0001):
		x2.append(i)
	
	#plt.plot(x2, np.exp(x1), color='red', label='exp(-x)')

	#plt.plot(x2, 0.5*np.exp(x3), color='green')
#print (np.mean(x))

# to code a counter to plot a manual histogram for the power after normalisation

	l=len(full1)

	a=Counter(full1[1:]).keys()
	b=Counter(full1[1:]).values()
	l2=l-1

	s=len(b)
	for i in range(0, s, 1):
		b[i]=b[i]/l2

	print (b[1:15])
	print ('mean nunorm power', np.mean(fourier_pow[100:]))
	print ('rms unnorm power', np.std(fourier_pow[100:]))
	print (max(fourier_pow))
	print (min(fourier_pow))

	print ('power mean ', np.mean(full[1:]))
	print('power std', np.std(full[1:]))
#plt.hist(full, b)
#plt.xlim(1700000000000, 53000000000000)
	#plt.xlim(0, 13)
#	plt.legend()
	#plt.yticks(np.arange(0, 1, 0.05))
	#plt.xticks(np.arange(0, 11.5, 0.5))
#	plt.xlabel('Fourier power')
#	plt.ylabel('Fourier power distributiom')
#	plt.title('red check, fp, point jump seg 3 1st seg, 2nd seg jump 5, seg 2nd point jump 5')
	#plt.title('Fourier Power probability distribution (post dereddening)- HTRU-N data')
	#`text('Area under histogram = 1')
	#print (bins)
	#plt.savefig('fourier_amp_bef_dered_1241_00.pdf')
#	plt.show()
	print (len(full))
	
	return(full1, fourier_freq_res)


def four_beam(p, p0, p1, p2, p3, p4, p5 ,p6):
	g=len(p)
#	for i in range(0, g, 1):
#		print ('elements sent to the are',p[i])
	c=c0=c1=c2=c3=c4=c5=c6=1
	if p[0]>=4.16 and p[1]>=4.16 and p[2]>=4.16 and p[3]>=4.16:
		c=1
		if p0 not in p:
			c0=0
		if p1 not in p:
			c1=0
		if p2 not in p:
			c2=0
		if p3 not in p:
			c3=0
		if p4 not in p:
			c4=0
		if p5 not in p:
			c5=0
		if p6 not in p:
			c6=0
			
		return c, c0, c1, c2, c3, c4, c5, c6
	else:
		c=0
		c1=0
		c2=0
		c3=0
		c4=0
		c5=0
		c6=0
		c0=0

		return c, c0, c1, c2, c3, c4, c5, c6

def five_beam(p, p0, p1, p2, p3, p4, p5 ,p6):

	g=len(p)
#	for i in range(0, g, 1):
#		print ('elements sent to the are',p[i])
	c=c0=c1=c2=c3=c4=c5=c6=1
	if p[0]>=3.33 and p[1]>=3.33 and p[2]>=3.33 and p[3]>=3.33 and p[4]>=3.33:	
		c=1
                if p0 not in p:
                        c0=0
                if p1 not in p:
                        c1=0
                if p2 not in p:
                        c2=0
                if p3 not in p:
                        c3=0
                if p4 not in p:
                        c4=0
                if p5 not in p:
                        c5=0
                if p6 not in p:
                        c6=0

                return c, c0, c1, c2, c3, c4, c5, c6
        else:
                c=0
                c1=0
                c2=0
                c3=0
                c4=0
                c5=0
                c6=0
                c0=0

                return c, c0, c1, c2, c3, c4, c5, c6
	
	
	
		


def six_beam(p, p0, p1, p2, p3, p4, p5 ,p6):
	g=len(p)
#	for i in range(0, g, 1):
#		print ('elements sent to the are',p[i])
	c=c0=c1=c2=c3=c4=c5=c6=1

	if p[0]>=2.773 and p[1]>=2.773 and p[2]>=2.773 and p[3]>=2.773 and p[4]>=2.773 and p[5]>=2.773:
		c=1
                if p0 not in p:
                        c0=0
                if p1 not in p:
                        c1=0
                if p2 not in p:
                        c2=0
                if p3 not in p:
                        c3=0
                if p4 not in p:
                        c4=0
                if p5 not in p:
                        c5=0
                if p6 not in p:
                        c6=0

                return c, c0, c1, c2, c3, c4, c5, c6
        else:
                c=0
                c1=0
                c2=0
                c3=0
                c4=0
                c5=0
                c6=0
                c0=0

                return c, c0, c1, c2, c3, c4, c5, c6




def seven_beam(p, p0, p1, p2, p3, p4, p5, p6):
	g=len(p)
#	for i in range(0, g, 1):
#		print ('elements sent to the are',p[i])
        c=c0=c1=c2=c3=c4=c5=c6=1

	if p[0]>=2.38 and p[1]>=2.38 and p[2]>=2.38 and p[3]>=2.38 and p[4]>=2.38 and p[5]>=2.38 and p[6]>=2.38:
		c=1
                if p0 not in p:
                        c0=0
                if p1 not in p:
                        c1=0
                if p2 not in p:
                        c2=0
                if p3 not in p:
                        c3=0
                if p4 not in p:
                        c4=0
                if p5 not in p:
                        c5=0
                if p6 not in p:
                        c6=0

                return c, c0, c1, c2, c3, c4, c5, c6
        else:
                c=0
                c1=0
                c2=0
                c3=0
                c4=0
                c5=0
                c6=0
                c0=0

                return c, c0, c1, c2, c3, c4, c5, c6


################################################# MAIN PROGRAM ####################################################################################################333333


if __name__ == '__main__':

	pointing=sys.argv[1]
#sending to the funtion to get the dereddened fourier_power with frew

	power0, freq= dered_norm_fp('00', pointing)	
#	plt.plot(freq[100:30000], power0[100:30000])
#	plt.show()
	power1, freq= dered_norm_fp('01', pointing)
        power2, freq= dered_norm_fp('02', pointing)          
	power3, freq= dered_norm_fp('03', pointing)
	power4, freq= dered_norm_fp('04', pointing)
	power5, freq= dered_norm_fp('05', pointing)
	power6, freq= dered_norm_fp('06', pointing)
	print ('length of the fnction received arrays are', len(power3), len(freq))
        #received the de reddened power series

	#===================== for 4 beam mitigation of threshold power in fourier_bins ===========================================

	z=len(freq)
#	g=l-2
	bad_freq=[]
	bf0=[]
	bf1=[]
	bf2=[]
	bf3=[]
	bf4=[]
	bf5=[]
	bf6=[]
	rf=freq[1]-freq[0]
	print ('freq res of received array is', rf)
	for i in range(1, z, 1):
		pa0 = power0[i]#+ power0[i+1] + power0[i+2])/3
		pa1 = power1[i]#+ power1[i+1] + power1[i+2])/3
		pa2 = power2[i]#+ power2[i+1] + power2[i+2])/3
		pa3 = power3[i]#+ power3[i+1] + power3[i+2])/3
		pa4 = power4[i]#+ power4[i+1] + power4[i+2])/3
		pa5 = power5[i]#+ power5[i+1] + power5[i+2])/3
		pa6 = power6[i]#+ power6[i+1] + power6[i+2])/3
		
		a=[pa0, pa1, pa2, pa3, pa4, pa5, pa6]
		b=list(itertools.combinations(a, 4))	
		
		counter=0
		counter0=0
		counter1=0
		counter2=0
		counter3=0
		counter4=0
		counter5=0
		counter6=0
#	        print ('freq is', freq[i], i)
		l=len(b)
		for m in range(0, l, 1):
			r, r0, r1, r2, r3, r4, r5, r6=four_beam(b[m], pa0, pa1, pa2, pa3, pa4, pa5, pa6)
			counter=counter+r
			counter0=counter0+r0
			counter1=counter1+r1
			counter2=counter2+r2
			counter3=counter3+r3
			counter4=counter4+r4
			counter5=counter5+r5
			counter6=counter6+r6
		#	print ('counter is', counter)
		d=1
		
		if counter==0:
			d=0
		else:
			f=freq[i]
			
#			print ('frequency is,', freq[i])
			if counter0>0:
				bf0.append(f)
			if counter1>0:
				bf1.append(f)
			if counter2>0:
				bf2.append(f)
			if counter3>0:
				bf3.append(f)
			if counter4>0:
				bf4.append(f)
			if counter5>0:
				bf5.append(f)
			if counter6>0:
				bf6.append(f)
			bad_freq.append(f) #storing the bad frequency bins in an array

	
		c1=list(itertools.combinations(a, 5))                 #combo for 5beam

		count=0
		count0=0
                count1=0
                count2=0
                count3=0
                count4=0
                count5=0
                count6=0

		l1=len(c1)
		for m1 in range(0, l1, 1):
			s, s0, s1, s2, s3, s4, s5, s6=five_beam(c1[m1], pa0, pa1, pa2, pa3, pa4, pa5, pa6)
                        count=count+s
                        count0=count0+s0
                        count1=count1+s1
                        count2=count2+s2
                        count3=count3+s3
                        count4=count4+s4
                        count5=count5+s5
                        count6=count6+s6

			

		if count==0:
			q='doing nothing'
		else:
			f=freq[i]
			if count0>0:
                                bf0.append(f)
                        if count1>0:
                                bf1.append(f)
                        if count2>0:
                                bf2.append(f)
                        if count3>0:
                                bf3.append(f)
                        if count4>0:
                                bf4.append(f)
                        if count5>0:
                                bf5.append(f)
                        if count6>0:
                                bf6.append(f)

			bad_freq.append(f)

		



		d=list(itertools.combinations(a, 6))		     #combo for 6beam

		coun=0
		coun1=0
		coun0=0
		coun2=0
		coun3=0
		coun4=0
		coun5=0
		coun6=0

		l2=len(d)
		for m2 in range(0, l2, 1):
			t, t0, t1, t2, t3, t4, t5, t6=six_beam(d[m2], pa0, pa1, pa2, pa3, pa4, pa5, pa6)
                        coun=coun+t
                        coun0=coun0+t0
                        coun1=coun1+t1
                        coun2=coun2+t2
                        coun3=coun3+t3
                        coun4=coun4+t4
                        coun5=coun5+t5
                        coun6=coun6+t6
			

		
		if coun==0:
			q='doing nothing'
		else:
			f=freq[i]
			bad_freq.append(f)
			if coun0>0:
                                bf0.append(f)
                        if coun1>0:
                                bf1.append(f)
                        if coun2>0:
                                bf2.append(f)
                        if coun3>0:
                                bf3.append(f)
                        if coun4>0:
                                bf4.append(f)
                        if coun5>0:
                                bf5.append(f)
                        if coun6>0:
                                bf6.append(f)




		e=list(itertools.combinations(a, 7))                 #combo for 7beam

		cou=0
		cou0=0
		cou1=0
		cou2=0
		cou3=0
		cou4=0
		cou5=0
		cou6=0
		l3=len(e)
		for m3 in range(0, l3, 1):
			u, u0, u1, u2, u3, u4, u5, u6=seven_beam(e[m3], pa0, pa1, pa2, pa3, pa4, pa5, pa6)
			cou=cou+u
                        cou0=cou0+u0
                        cou1=cou1+u1
                        cou2=cou2+u2
                        cou3=cou3+u3
                        cou4=cou4+u4
                        cou5=cou5+u5
                        cou6=cou6+u6

		

	
			

		if cou==0:
			q='doing nothing'
		else:
			f=freq[i]
			bad_freq.append(f)
			if cou0>0:
                                bf0.append(f)
                        if cou1>0:
                                bf1.append(f)
                        if cou2>0:
                                bf2.append(f)
                        if cou3>0:
                                bf3.append(f)
                        if cou4>0:
                                bf4.append(f)
                        if cou5>0:
                                bf5.append(f)
                        if cou6>0:
                                bf6.append(f)














	x=[]
	m=len(bad_freq)
	for k in bad_freq:
		if k not in x:
			x.append(k)

	
	x0=[]
        m0=len(bf0)
        for k in bf0:
                if k not in x0:
                        x0.append(k)
 
	x1=[]
        m1=len(bf1)
        for k in bf1:
                if k not in x1:
                        x1.append(k)

	x2=[]
        m2=len(bf2)
        for k in bf2:
                if k not in x2:
                        x2.append(k)

	x3=[]
        m3=len(bf3)
        for k in bf3:
                if k not in x3:
                        x3.append(k)

	x4=[]
        m4=len(bf4)
        for k in bf4:
                if k not in x4:
                        x4.append(k)

	x5=[]
        m5=len(bf5)
        for k in bf5:
                if k not in x5:
                        x5.append(k)

	x6=[]
        m6=len(bf6)
        for k in bf6:
                if k not in x6:
                        x6.append(k)



	print ('the bad frequency bins are:', x)
	print ('length of the bad freq bins is,', len(x))
	y=len(b)
	o='/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/filterbank_files/' 
	q=o+str(pointing)+'_zaplist_bs.zaplist'
	f=open(q,'w')
	f.write('#Freq \t\t\t\t Width \t\t\t #harm \t\t grow? \t\t bary?\n')
	k=len(x)
	d=(freq[1]-freq[0])*2
	for i in range(0, k, 1):
		if x[i]<=1830:
			f.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x[i], d, 0, 0, 0))
		if x[i]>1830 and x[i]<=2000:
			f.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x[i], d, 0, 0, 0))
		if x[i]>2000 and x[i]<=3000:
			f.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x[i], d, 0, 0, 0))
		if x[i]>3000 and x[i]<=4500:
			f.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x[i], d, 0, 0, 0))
		if x[i]>4500:
			f.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x[i], d, 0, 0, 0))
	f.close()


	w=o+str(pointing)+'_zaplist_bs_nowid_yesharm.zaplist'
	g=open(w,'w')
        g.write('#Freq \t\t\t\t Width \t\t #harm \t\t grow? \t\t bary?\n')
        k=len(x)
        d=(freq[1]-freq[0])*2
        for i in range(0, k, 1):
                if x[i]<=1830:
                        g.write("%24.18f \t %4d \t\t %4d \t\t %4d \t\t %4d\n"%(x[i], 0, 5, 0, 0))
                if x[i]>1830 and x[i]<=2000:
                        g.write("%24.18f \t %4d \t\t %4d \t\t %4d \t\t %4d\n"%(x[i], 0, 4, 0, 0))
                if x[i]>2000 and x[i]<=3000:
                        g.write("%24.18f \t %4d \t\t %4d \t\t %4d \t\t %4d\n"%(x[i], 0, 3, 0, 0))
                if x[i]>3000 and x[i]<=4500:
                        g.write("%24.18f \t %4d \t\t %4d \t\t %4d \t\t %4d\n"%(x[i], 0, 2, 0, 0))
                if x[i]>4500:
                        g.write("%24.18f \t %4d \t\t %4d \t\t %4d \t\t %4d\n"%(x[i], 0, 1, 0, 0))
	g.close()

	w0='/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/filterbank_files/' + 'beam00/'
	q0=w0+str(pointing)+'_00_zaplist_bs.zaplist'
	g0=open(q0,'w')
        g0.write('#Freq \t\t\t\t Width \t\t\t #harm \t\t grow? \t\t bary?\n')
        k=len(x0)
        d=(freq[1]-freq[0])*2
        for i in range(0, k, 1):
                if x0[i]<=1830:
                        g0.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x0[i], d, 0, 0, 0))
                if x0[i]>1830 and x0[i]<=2000:
                        g0.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x0[i], d, 0, 0, 0))
                if x0[i]>2000 and x0[i]<=3000:
                        g0.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x0[i], d, 0, 0, 0))
                if x0[i]>3000 and x0[i]<=4500:
                        g0.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x0[i], d, 0, 0, 0))
                if x0[i]>4500:
                        g0.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x0[i], d, 0, 0, 0))


        g0.close()

	w1='/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/filterbank_files/' + 'beam01/'
        q1=w1+str(pointing)+'_01_zaplist_bs.zaplist'

	g1=open(q1,'w')
        g1.write('#Freq \t\t\t\t Width \t\t\t #harm \t\t grow? \t\t bary?\n')
        k=len(x1)
        d=(freq[1]-freq[0])*2
        for i in range(0, k, 1):
                if x1[i]<=1830:
                        g1.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x1[i], d, 0, 0, 0))
                if x1[i]>1830 and x1[i]<=2000:
                        g1.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x1[i], d, 0, 0, 0))
                if x1[i]>2000 and x1[i]<=3000:
                        g1.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x1[i], d, 0, 0, 0))
                if x1[i]>3000 and x1[i]<=4500:
                        g1.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x1[i], d, 0, 0, 0))
                if x1[i]>4500:
                        g1.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x1[i], d, 0, 0, 0))


        g1.close()
	

	w2='/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/filterbank_files/' + 'beam02/'
        q2=w2+str(pointing)+'_02_zaplist_bs.zaplist'

	g2=open(q2,'w')
        g2.write('#Freq \t\t\t\t Width \t\t\t #harm \t\t grow? \t\t bary?\n')
        k=len(x2)
        d=(freq[1]-freq[0])*2
        for i in range(0, k, 1):
                if x2[i]<=1830:
                        g2.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x2[i], d, 0, 0, 0))
                if x2[i]>1830 and x2[i]<=2000:
                        g2.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x2[i], d, 0, 0, 0))
                if x2[i]>2000 and x2[i]<=3000:
                        g2.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x2[i], d, 0, 0, 0))
                if x2[i]>3000 and x2[i]<=4500:
                        g2.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x2[i], d, 0, 0, 0))
                if x2[i]>4500:
                        g2.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x2[i], d, 0, 0, 0))


        g2.close()


	w3='/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/filterbank_files/' + 'beam03/'
        q3=w3+str(pointing)+'_03_zaplist_bs.zaplist'

	g3=open(q3,'w')
        g3.write('#Freq \t\t\t\t Width \t\t\t #harm \t\t grow? \t\t bary?\n')
        k=len(x3)
        d=(freq[1]-freq[0])*2
        for i in range(0, k, 1):
                if x3[i]<=1830:
                        g3.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x3[i], d, 0, 0, 0))
                if x3[i]>1830 and x3[i]<=2000:
                        g3.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x3[i], d, 0, 0, 0))
                if x3[i]>2000 and x3[i]<=3000:
                        g3.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x3[i], d, 0, 0, 0))
                if x3[i]>3000 and x3[i]<=4500:
                        g3.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x3[i], d, 0, 0, 0))
                if x3[i]>4500:
                        g3.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x3[i], d, 0, 0, 0))


        g3.close()



	w4='/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/filterbank_files/' + 'beam04/'
        q4=w4+str(pointing)+'_04_zaplist_bs.zaplist'

	g4=open(q4,'w')
        g4.write('#Freq \t\t\t\t Width \t\t\t #harm \t\t grow? \t\t bary?\n')
        k=len(x4)
        d=(freq[1]-freq[0])*2
        for i in range(0, k, 1):
                if x4[i]<=1830:
                        g4.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x4[i], d, 0, 0, 0))
                if x4[i]>1830 and x4[i]<=2000:
                        g4.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x4[i], d, 0, 0, 0))
                if x4[i]>2000 and x4[i]<=3000:
                        g4.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x4[i], d, 0, 0, 0))
                if x4[i]>3000 and x4[i]<=4500:
                        g4.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x4[i], d, 0, 0, 0))
                if x4[i]>4500:
                        g4.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x4[i], d, 0, 0, 0))


        g4.close()

	

	w5='/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/filterbank_files/' + 'beam05/'
        q5=w5+str(pointing)+'_05_zaplist_bs.zaplist'

	g5=open(q5,'w')
        g5.write('#Freq \t\t\t\t Width \t\t\t #harm \t\t grow? \t\t bary?\n')
        k=len(x5)
        d=(freq[1]-freq[0])*2
        for i in range(0, k, 1):
                if x5[i]<=1830:
                        g5.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x5[i], d, 0, 0, 0))
                if x5[i]>1830 and x5[i]<=2000:
                        g5.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x5[i], d, 0, 0, 0))
                if x5[i]>2000 and x5[i]<=3000:
                        g5.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x5[i], d, 0, 0, 0))
                if x5[i]>3000 and x5[i]<=4500:
                        g5.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x5[i], d, 0, 0, 0))
                if x5[i]>4500:
                        g5.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x5[i], d, 0, 0, 0))


        g5.close()



	w6='/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/filterbank_files/' + 'beam06/'
        q6=w6+str(pointing)+'_06_zaplist_bs.zaplist'

	g6=open(q6,'w')
        g6.write('#Freq \t\t\t\t Width \t\t\t #harm \t\t grow? \t\t bary?\n')
        k=len(x6)
        d=(freq[1]-freq[0])*2
        for i in range(0, k, 1):
                if x6[i]<=1830:
                        g6.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x6[i], d, 0, 0, 0))
                if x6[i]>1830 and x6[i]<=2000:
                        g6.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x6[i], d, 0, 0, 0))
                if x6[i]>2000 and x6[i]<=3000:
                        g6.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x6[i], d, 0, 0, 0))
                if x6[i]>3000 and x6[i]<=4500:
                        g6.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x6[i], d, 0, 0, 0))
                if x6[i]>4500:
                        g6.write("%24.18f \t %12.10f \t\t %4d \t\t %4d \t\t %4d\n"%(x6[i], d, 0, 0, 0))


        g6.close()









	'''with open('1241_bad_freq_all_beam_new_res.txt', 'w') as f:
		for item in x:
			f.write("%s\n" % item)'''
	for s in range(0, y, 1):
		print (b[s])
#	plt.plot(power4[1:],freq[1:])
#	plt.show()
        
	print (r)

