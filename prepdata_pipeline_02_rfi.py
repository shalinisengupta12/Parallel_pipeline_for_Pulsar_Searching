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



#============================================================= FUNCTIONS ============================================================================



def split_data_filterbank(pointing, beam_number, output_location):
#directory where the raw filterbanks are kept, different from where they will be dumped after segment
#	beam='beam%s' %str(beam_number)
#	os.chdir(beam)
	#filterbank_data = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/filterbank_files/' + 'beam%s' %str(beam_number) + '/' + pointing + '_' + str(beam_number) + '_8bit'+ '.fil'
    filterbank_data = pointing + '_' + str(beam_number) + '_8bit' +  '.fil'
	
    input_file1 = FilReader(filterbank_data)
    SKIP = 0
    input_file1.split(0, 33554432, filename= pointing + '_' + str(beam_number) + '_2powerd4' +  '.fil')
    filterbank_corrected = pointing + '_' + str(beam_number) + '_2powerd4' +  '.fil'	
    input_file2 = FilReader(filterbank_corrected)
    filterbank = pointing + '_' + str(beam_number) + '_2power' +  '.fil'
    input_file2.downsample(tfactor=4, ffactor=1, filename=filterbank)
    input_file = FilReader(filterbank)
    number_samples = input_file.header.nsamples
    obs_time_seconds = input_file.header.tobs
    sampling_time = input_file.header.tsamp
#	cmds='touch fil_list.txt'
#	input_file.header.telescope_id = 4
    for i in range(2, 6, 2):
        j=i
        NTOREAD = int(int(number_samples)/i)
        for k in range(0, j, 1):
            num_samp= (k*NTOREAD)
            input_file.split(num_samp, NTOREAD, filename = output_location +  '/' + pointing + '_beam_%s' %str(beam_number) + '_segment_By_%s' %str(i) + '_part_%s' %str(k) + '.fil')

    #copying the filterbanks to the home area, will be used llater for each's individual searches
    cmds = 'cp ' + filterbank + ' ' + output_location
    log = subprocess.check_output(cmds,shell=True)
    print log
    head_dir = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poiting'





def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5\\
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise




def rfi_find(pointing_name, beam_number):

  #  current_filterbank = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing_name +'/filterbank_files/' +  'beam%s/' %str(beam_number) + pointing_name + '_%s_' %str(beam_number) + '2power.fil'
    current_filterbank = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing_name +'/filterbank_files/' +  'beam%s/' %str(beam_number) + pointing_name + '_%s_' %str(beam_number) + '8bit.fil' 
    current_filterbank2 = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing_name +'/filterbank_files/' +  'beam%s/' %str(beam_number) + pointing_name + '_%s_' %str(beam_number) + '2power.fil'
    cmds = 'rfifind -ncpus 24 -filterbank -rfips -time 15.0 -zerodm -zapchan 63:66,88:150,415:447,190,291:293 -o %s %s' %(os.path.basename(current_filterbank)[:-9], current_filterbank)
    log = subprocess.check_output(cmds, shell=True)
    print log
    cmds = 'rfifind -ncpus 24 -filterbank -rfips -time 15.0 -zerodm -zapchan 63:66,88:150,415:447,190,291:293 -o %s %s' %(os.path.basename(current_filterbank2)[:-4], current_filterbank2)
    log = subprocess.check_output(cmds, shell=True)
    print log






def rfi_find_segmented(pointing_name, beam_number, segment_number, part_number):

    if segment_number==2:
        if part_number==0:
            current_filterbank = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing_name +'/filterbank_files/' +  'beam%s/' %str(beam_number) + pointing_name + '_beam_' + str(beam_number) + '_segment_By_' + str(segment_number) + '_part_' + str(part_number) + '.fil'
            cmds = 'rfifind -ncpus 24 -filterbank -rfips -time 10 -zerodm -intfrac 0.4 -zapchan 63:66,88:150,415:447,190,291:293 -o %s %s' %(os.path.basename(current_filterbank)[:-4], current_filterbank)
            log = subprocess.check_output(cmds, shell=True)
            print log
        else:
            current_filterbank = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing_name +'/filterbank_files/' +  'beam%s/' %str(beam_number) + pointing_name + '_beam_' + str(beam_number) + '_segment_By_' + str(segment_number) + '_part_' + str(part_number) + '.fil'
            cmds = 'rfifind -ncpus 24 -filterbank -rfips -time 10 -zerodm -intfrac 0.4 -zapchan 63:66,88:150,415:447,190,291:293 -o %s %s' %(os.path.basename(current_filterbank)[:-4], current_filterbank)
            log = subprocess.check_output(cmds, shell=True)


    if segment_number==4:
        if part_number==0:
            current_filterbank = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing_name +'/filterbank_files/' +  'beam%s/' %str(beam_number) + pointing_name + '_beam_' + str(beam_number) + '_segment_By_' + str(segment_number) + '_part_' + str(part_number) + '.fil'
            cmds = 'rfifind -ncpus 24 -filterbank -rfips -time 5 -zerodm -intfrac 0.5 -zapchan 63:66,88:150,415:447,190,291:293 -o %s %s' %(os.path.basename(current_filterbank)[:-4], current_filterbank)
            log = subprocess.check_output(cmds, shell=True)
            print log
        elif part_number==1:
            current_filterbank = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing_name +'/filterbank_files/' +  'beam%s/' %str(beam_number) + pointing_name + '_beam_' + str(beam_number) + '_segment_By_' + str(segment_number) + '_part_' + str(part_number) + '.fil'
            cmds = 'rfifind -ncpus 24 -filterbank -rfips -time 5 -zerodm -intfrac 0.5 -zapchan 63:66,88:150,415:447,190,291:293 -o %s %s' %(os.path.basename(current_filterbank)[:-4], current_filterbank)
            log = subprocess.check_output(cmds, shell=True)
        
        elif part_number==2:
            current_filterbank = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing_name +'/filterbank_files/' +  'beam%s/' %str(beam_number) + pointing_name + '_beam_' + str(beam_number) + '_segment_By_' + str(segment_number) + '_part_' + str(part_number) + '.fil'
            cmds = 'rfifind -ncpus 24 -filterbank -rfips -time 5 -zerodm -intfrac 0.5 -zapchan 63:66,88:150,415:447,291:293,190 -o %s %s' %(os.path.basename(current_filterbank)[:-4], current_filterbank)
            log = subprocess.check_output(cmds, shell=True)
            print log
        else:
            current_filterbank = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing_name +'/filterbank_files/' +  'beam%s/' %str(beam_number) + pointing_name + '_beam_' + str(beam_number) + '_segment_By_' + str(segment_number) + '_part_' + str(part_number) + '.fil'
            cmds = 'rfifind -ncpus 24 -filterbank -rfips -time 5 -zerodm -intfrac 0.5 -zapchan 63:66,88:150,415:447,291:293,190 -o %s %s' %(os.path.basename(current_filterbank)[:-4], current_filterbank)
            log = subprocess.check_output(cmds, shell=True)







def dedisperse_batch1(filename, number_samples, mask_file, dm):
    if dm < 14.126:
        output_filename =  os.path.basename(filename[:-4]) + '_DM_%s' %str(dm) + '_d1a'
        DM = str(dm)
        cmds = 'prepdata -o %s -filterbank -numout %s -downsamp 1 -dm %s -mask %s %s' %(output_filename, number_samples, DM, mask_file, filename)
        log = subprocess.check_output(cmds,shell=True)
        print log
    else:
        output_filename =  os.path.basename(filename[:-4]) + '_DM_%s' %str(dm) + '_d1b'
        DM = str(dm)
        cmds = 'prepdata -o %s -filterbank -numout %s -downsamp 1 -dm %s -mask %s %s' %(output_filename, number_samples, DM, mask_file, filename)
        log = subprocess.check_output(cmds,shell=True)
        print log

def dedisperse_batch2(filename, number_samples, mask_file, dm):
#    for dm in np.arange(113.0095, 226.0170, 0.442):
    output_filename =  os.path.basename(filename[:-4]) + '_DM_%s' %str(dm) + '_d2'
    DM = str(dm)
    cmds = 'prepdata -o %s -filterbank -numout %s -downsamp 2 -dm %s -mask %s %s' %(output_filename, number_samples, DM, mask_file, filename)
    log = subprocess.check_output(cmds,shell=True)
    print log

def dedisperse_batch3(filename, number_samples, mask_file, dm):
   # for dm in np.arange(226.0170, 452.034, 0.884):
    output_filename =  os.path.basename(filename[:-4]) + '_DM_%s' %str(dm) + '_d3'
    DM = str(dm)
    cmds = 'prepdata -o %s -filterbank -zerodm -numout %s -downsamp 4 -dm %s -mask %s %s' %(output_filename, number_samples, DM, mask_file, filename)
    log = subprocess.check_output(cmds,shell=True)
    print log

def dedisperse_batch4(filename, number_samples, mask_file, dm):
  #  for dm in np.arange(452.034, 904.068, 1.766):
    output_filename =  os.path.basename(filename[:-4]) + '_DM_%s' %str(dm) + '_d4'
    DM = str(dm)
    cmds = 'prepdata -o %s -filterbank -zerodm -numout %s -downsamp 4 -dm %s -mask %s %s' %(output_filename, number_samples, DM, mask_file, filename)
    log = subprocess.check_output(cmds,shell=True)
    print log

def dedisperse_batch5(filename, number_samples, mask_file, dm):
   # for dm in np.arange(904.068, 1808.036, 3.53):
    output_filename =  os.path.basename(filename[:-4]) + '_DM_%s' %str(dm) + '_d5'
    DM = str(dm)
    cmds = 'prepdata -o %s -filterbank -zerodm -numout %s -downsamp 4 -dm %s -mask %s %s' %(output_filename, number_samples, DM, mask_file, filename)
    log = subprocess.check_output(cmds,shell=True)


def dedisperse_batch6(filename, number_samples, mask_file, dm):
   # for dm in np.arange(1808.136, 3001.783, 7.063):
    output_filename =  os.path.basename(filename[:-4]) + '_DM_%s' %str(dm) + '_d6'
    DM = str(dm)
    cmds = 'prepdata -o %s -filterbank -zerodm -numout %s -downsamp 8 -dm %s -mask %s %s' %(output_filename, number_samples, DM, mask_file, filename)
    log = subprocess.check_output(cmds,shell=True)
    print log

def dedisperse_batch7(filename, number_samples, mask_file, dm):
   # for dm in np.arange(1808.136, 3001.783, 7.063):
    output_filename =  os.path.basename(filename[:-4]) + '_DM_%s' %str(dm) + '_d7'
    DM = str(dm)
    cmds = 'prepdata -o %s -filterbank -zerodm -numout %s -downsamp 16 -dm %s -mask %s %s' %(output_filename, number_samples, DM, mask_file, filename)
    log = subprocess.check_output(cmds,shell=True)
    print log

def dedisperse_batch8(filename, number_samples, mask_file, dm):
   # for dm in np.arange(1808.136, 3001.783, 7.063):
    output_filename =  os.path.basename(filename[:-4]) + '_DM_%s' %str(dm) + '_d8'
    DM = str(dm)
    cmds = 'prepdata -o %s -filterbank -zerodm -numout %s -downsamp 32 -dm %s -mask %s %s' %(output_filename, number_samples, DM, mask_file, filename)
    log = subprocess.check_output(cmds,shell=True)
    print log






def accelsearch(filename, zapfile2):
   # input_filename = filename + str(i) + '*.dat'
    cmds = 'accelsearch -zaplist ' + zapfile2 + ' -numharm 16 -zmax 100 ' + filename
    log = subprocess.check_output(cmds,shell=True)
    print log



def correct_header(filename_inf, root_name):
    k =  re.findall("\d+\.\d+", filename_inf)
    k1 = k[0]
    k2= float(k1)
    k3= "%.2f" %k2
    cmds = 'mv ' + filename_inf + ' ' + root_name + '_DM%s.inf' %str(k3)
    log = subprocess.check_output(cmds,shell=True)
    print log

def correct_ACC(filename_ACC, root_name):
    g = re.findall("\d+\.\d+", filename_ACC)
    g1= g[0]
    g2= float(g1)
    g3 = "%.2f" %g2
    cmds = 'mv ' + filename_ACC + ' ' + root_name + '_DM%s_ACCEL_100' %str(g3)
    log = subprocess.check_output(cmds,shell=True)
    print log
    cmds = 'mv ' + filename_ACC + '.cand ' + root_name + '_DM%s_ACCEL_100.cand' %str(g3)
    log = subprocess.check_output(cmds,shell=True)
    print log










def fold1(a, b, filterbank, mask, plot_folder, processing_folder):
    os.chdir(processing_folder)
    k =  re.findall("\d+\.\d+", a)
    print k
    k1 = k[0]
    print k1
    a3= a[:-10] 
    print a3
   # a2= plot_folder + '/' + a3
    a4= a+'.cand'
    cmds = 'prepfold -o %s -noxwin -dm %s -accelcand %s -accelfile %s -mask %s %s' %(a3, k1, b, a4, mask, filterbank)
    log = subprocess.check_output(cmds,shell=True)
    print log






# ======================================================== FUNCTIONS FOR THE SEGMENTED PARTS : DEDISPERSION 1ST =====================

def Sdedisperse_batch1(filename1, number_samples, mask_file, dm, processing_folder):
    os.chdir(processing_folder)
    if dm < 56.4:
        output_filename =  processing_folder + os.path.basename(filename1[:-4]) + '_DM_%s' %str(dm) +'_d1a'
        filename = processing_folder + filename1
        DM = str(dm)
        cmds = 'prepdata -o %s -filterbank -numout %s -downsamp 1 -dm %s -mask %s %s' %(output_filename, number_samples, DM, mask_file, filename)
        log = subprocess.check_output(cmds,shell=True)
        print log
    else:
        output_filename =  processing_folder + os.path.basename(filename1[:-4]) + '_DM_%s' %str(dm) +'_d1b'
        filename = processing_folder + filename1
        DM = str(dm)
        cmds = 'prepdata -o %s -filterbank -zerodm -numout %s -downsamp 1 -dm %s -mask %s %s' %(output_filename, number_samples, DM, mask_file, filename)
        log = subprocess.check_output(cmds,shell=True)
        print log

def Sdedisperse_batch2(filename1, number_samples, mask_file, dm, processing_folder):
 #   for dm in np.arange(113.0095, 114.894, 0.442):
    os.chdir(processing_folder)
    output_filename =  processing_folder + os.path.basename(filename1[:-4]) + '_DM_%s' %str(dm) + '_d2'
    filename = processing_folder + filename1
    DM = str(dm)
    cmds = 'prepdata -o %s -filterbank -zerodm -numout %s -downsamp 1 -dm %s -mask %s %s' %(output_filename, number_samples, DM, mask_file, filename)
    log = subprocess.check_output(cmds,shell=True)
    print log

def Sdedisperse_batch3(filename1, number_samples, mask_file, dm, processing_folder):
  #  for dm in np.arange(226.0170, 227.785, 0.884):
    os.chdir(processing_folder)
    output_filename = processing_folder + os.path.basename(filename1[:-4]) + '_DM_%s' %str(dm) + '_d3'
    filename = processing_folder + filename1
    DM = str(dm)
    cmds = 'prepdata -o %s -filterbank -zerodm -numout %s -downsamp 1 -dm %s -mask %s %s' %(output_filename, number_samples, DM, mask_file, filename)
    log = subprocess.check_output(cmds,shell=True)
    print log

def Sdedisperse_batch4(filename1, number_samples, mask_file, dm, processing_folder):
  #  for dm in np.arange(452.034, 904.068, 1.766):
    os.chdir(processing_folder)
    output_filename = processing_folder +  os.path.basename(filename1[:-4]) + '_DM_%s' %str(dm) + '_d4'
    filename = processing_folder + filename1
    DM = str(dm)
    cmds = 'prepdata -o %s -filterbank -zerodm -numout %s -downsamp 2 -dm %s -mask %s %s' %(output_filename, number_samples, DM, mask_file, filename)
    log = subprocess.check_output(cmds,shell=True)
    print log

def Sdedisperse_batch5(filename1, number_samples, mask_file, dm, processing_folder):
   # for dm in np.arange(904.068, 1808.036, 3.53):
    os.chdir(processing_folder)
    output_filename =  processing_folder + os.path.basename(filename1[:-4]) + '_DM_%s' %str(dm) + '_d5'
    filename = processing_folder + filename1
    DM = str(dm)
    cmds = 'prepdata -o %s -filterbank -zerodm -numout %s -downsamp 4 -dm %s -mask %s %s' %(output_filename, number_samples, DM, mask_file, filename)
    log = subprocess.check_output(cmds,shell=True)
    print log

def Sdedisperse_batch6(filename1, number_samples, mask_file, dm, processing_folder):
    os.chdir(processing_folder)
   # for dm in np.arange(1808.136, 3001.783, 7.063):
    output_filename = processing_folder +  os.path.basename(filename1[:-4]) + '_DM_%s' %str(dm) + '_d6'
    filename = processing_folder + filename1
    DM = str(dm)
    cmds = 'prepdata -o %s -filterbank -zerodm -numout %s -downsamp 8 -dm %s -mask %s %s' %(output_filename, number_samples, DM, mask_file, filename)
    log = subprocess.check_output(cmds,shell=True)
    print log







def accelsearch_S2(filename1, processing_folder, zapfile3):
    os.chdir(processing_folder)
    filename = processing_folder + filename1
    zap = processing_folder + zapfile3
   # input_filename = filename + str(i) + '*.dat'
    cmds = 'accelsearch -zaplist ' + zap+ ' -numharm 16 -zmax 667 ' + filename
    log = subprocess.check_output(cmds,shell=True)
    print log




def Scorrect_header(filename_inf1, root_name, processing_folder):
    os.chdir(processing_folder)
    file1 = processing_folder + filename_inf1
    filename_inf= os.path.basename(file1)
    k =  re.findall("\d+\.\d+", filename_inf)
    k1 = k[0]
    k2= float(k1)
    k3= "%.2f" %k2
    cmds = 'mv ' + filename_inf + ' ' + root_name + '_DM%s.inf' %str(k3)
    log = subprocess.check_output(cmds,shell=True)
    print log

def Scorrect_ACC(filename_ACC1, root_name, processing_folder):
    os.chdir(processing_folder)
    file1 = processing_folder + filename_ACC1
    filename_ACC= os.path.basename(file1)
    g = re.findall("\d+\.\d+", filename_ACC)
    g1= g[0]
    g2= float(g1)
    g3 = "%.2f" %g2
    cmds = 'mv ' + filename_ACC + ' ' + root_name + '_DM%s_ACCEL_668' %str(g3)
    log = subprocess.check_output(cmds,shell=True)
    print log
    cmds = 'mv ' + filename_ACC + '.cand ' + root_name + '_DM%s_ACCEL_668.cand' %str(g3)
    log = subprocess.check_output(cmds,shell=True)
    print log

def accelsearch_S4(filename1, processing_folder, zapfile4):
   # input_filename = filename + str(i) + '*.dat'
    os.chdir(processing_folder)
    filename = processing_folder + filename1
    zap =  processing_folder +zapfile4
    cmds = 'accelsearch -zaplist ' + zap + ' -numharm 16 -zmax 422 ' + filename
    log = subprocess.check_output(cmds,shell=True)
    print log


def S4correct_header(filename_inf1, root_name, processing_folder):
    os.chdir(processing_folder)
    file1 = processing_folder + filename_inf1
    filename_inf= os.path.basename(file1)
    k =  re.findall("\d+\.\d+", filename_inf)
    k1 = k[0]
    k2= float(k1)
    k3= "%.2f" %k2
    cmds = 'mv ' + filename_inf + ' ' + root_name + '_DM%s.inf' %str(k3)
    log = subprocess.check_output(cmds,shell=True)
    print log

def S4correct_ACC(filename_ACC1, root_name, processing_folder):
    os.chdir(processing_folder)
    file1 = processing_folder + filename_ACC1
    filename_ACC= os.path.basename(file1)
    g = re.findall("\d+\.\d+", filename_ACC)
    g1= g[0]
    g2= float(g1)
    g3 = "%.2f" %g2
    cmds = 'mv ' + filename_ACC + ' ' + root_name + '_DM%s_ACCEL_422' %str(g3)
    log = subprocess.check_output(cmds,shell=True)
    print log
    cmds = 'mv ' + filename_ACC + '.cand ' + root_name + '_DM%s_ACCEL_422.cand' %str(g3)
    log = subprocess.check_output(cmds,shell=True)
    print log












#============================================================  SPLITTING  THE  FILTERBANK  ======================================================================================================





if __name__ == '__main__':

    tfull= time.time()	
    pointing = sys.argv[1]
    beam = sys.argv[2]
    cmds = 'rm -rf /tmp/data_shalini'
    subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
#	beams = ['00','01','02','03','04','05','06']
    output_location = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/filterbank_files/beam%s' %str(beam)
    mkdir_p(output_location)
#	os.makedirs()
#	processing_folder1 = '/dev/shm/data_shalini/'
    processing_folder='/tmp/data_shalini/'
	
    os.makedirs(processing_folder)
    os.chdir(processing_folder)
#	for b in range(len(beams)):
#		x=beams[b]
    filterbank='/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/filterbank_files/' + 'beam%s/' %str(beam) + pointing + '_%s_' %str(beam) + '8bit.fil'
    cmds = 'cp ' + filterbank + ' ' + processing_folder
    subprocess.check_output(cmds,shell=True)
#	os.chdir(output_location)	
    t0 = time.time()
    #for beam_number in range(len(beams)):
    #    split_data_filterbank(pointing, fraction, beams[beam_number], output_location)

#	for k in range(len(beams)):
#		split_data_filterbank(pointing, fraction, beams[k], output_location) 
#	val = Parallel(n_jobs=num_cores)(delayed(split_data_filterbank)(pointing, fraction, beams[beam_number], output_location) for beam_number in range(len(beams)))
    split_data_filterbank(pointing, beam, output_location)
    cmds = 'rm -rf ' + processing_folder
    subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
    t1 = time.time()
    total_n = (t1-t0)
    count = 0
    print 'Time taken for the splitting_filterbank to execute was %s seconds' %(total_n)



#==========================================   RFIFINDING UNSEGMENTED FIRST THEN THE SPLIT FILES ==========================================================================



    
    results_location = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/rfi_masks'
    mkdir_p(results_location)
    os.chdir(results_location)
    t0 = time.time()
    #Parallel(n_jobs=num_cores)(delayed(rfi_find)(pointing_name, beam_number, segment_number + 1) for segment_number in range(fraction))

    rfi_find(pointing, beam)
    #cmds = 'rm -rf /dev/shm/data_vishnu'
    #subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
    t1 = time.time()
    total_n = (t1-t0)
    print 'Time taken for the code to complete RFI UNSEGMENTED  was %s seconds' %(total_n)



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& unsegmented rfi masks made. now will do the SEGMENTED part ========================================================

    results_location = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/rfi_masks'
    mkdir_p(results_location)
    os.chdir(results_location)
    t0 = time.time()
    #Parallel(n_jobs=num_cores)(delayed(rfi_find)(pointing_name, beam_number, segment_number + 1) for segment_number in range(fraction))
    for segment_number in range(2, 6, 2):
        a=segment_number
        for part_number in range(0, a, 1):
            rfi_find_segmented(pointing, beam, segment_number, part_number)
    #cmds = 'rm -rf /dev/shm/data_vishnu'
    #subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
    t1 = time.time()
    total_n = (t1-t0)
    print 'Time taken for the code to complete SEGMENTED ALL RFIFINDING %s seconds' %(total_n)


#======================================================== DEDISPERSING THE UNSEGMENTED FULL LENGTH FULL RESOLUTION FILTERBANK ====================================================


    t3 = time.time()
    processing_folder = '/tmp/data_shalini/'
    current_filterbank = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing +'/filterbank_files/' +  'beam%s/' %str(beam) + pointing + '_%s_' %str(beam) + '8bit.fil'
   # current_filterbank1 = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing_name +'/filterbank_files/' +  'beam%s/' %str(beam_number) + pointing_name + '_%s_' %str(beam_number) + '2power.fil'
    root_filterbank = os.path.basename(current_filterbank[:-9])
    mask_file = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/'+ pointing + '/rfi_masks' + '/' + root_filterbank + '_rfifind.mask'
   # results_location = '/home/psr/full_kepler_search/cpu_pipeline/TIMESERIES/' + pointing_name + '/'
    bytemask = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/'+ pointing + '/rfi_masks' + '/' + root_filterbank + '_rfifind.bytemask'
    ps = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/'+ pointing + '/rfi_masks' + '/' + root_filterbank + '_rfifind.ps'
    stats= '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/'+ pointing + '/rfi_masks' + '/' + root_filterbank + '_rfifind.stats'
    rfi = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/'+ pointing + '/rfi_masks' + '/' + root_filterbank + '_rfifind.rfi'
    zaplist='/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing +'/filterbank_files/' +  'beam%s/' %str(beam) + pointing + '_%s_' %str(beam) + 'zaplist_bs.zaplist'
#    results_location = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/Timeseries/'
 #   mkdir_p(results_location)
  #  results_location2= '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/Timeseries/' + beam
   # mkdir_p(results_location2)
    #results_location3 = results_location2 + '/' + 'Unsegmented'
  #  mkdir_p(results_location3)
  #  results_location4 = results_location3 + '/' + 'part%s' %str(timeseries_segment)
  #  mkdir_p(results_location4)
    cmds = 'rm -rf ' + processing_folder
    subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
    #processing_folder = '/tmp/data_vishnu/' + pointing_name + '/' + beam_number + '/' + timeseries_segment + '/' + str(dedisperse_batch) + '/'
    #processing_folder = '/tmp/data_vishnu/'
    mkdir_p(processing_folder)
    os.chdir(processing_folder)

    ''' Copy filterbank and rfi mask to processing directory and then run the code '''

    if not os.path.exists(processing_folder + current_filterbank):
        cmds = 'cp ' + current_filterbank + ' ' + processing_folder
        subprocess.check_output(cmds,shell=True)

#    if not os.path.exists(processing_folder + current_filterbank1):
 #       cmds = 'cp ' + current_filterbank1 + ' ' + processing_folder
 #       subprocess.check_output(cmds,shell=True)

    if not os.path.exists(processing_folder + mask_file):
        cmds = 'cp ' + mask_file + ' ' + processing_folder
        subprocess.check_output(cmds,shell=True)

    if not os.path.exists(processing_folder + bytemask):
        cmds = 'cp ' + bytemask + ' ' + processing_folder
        subprocess.check_output(cmds,shell=True)

    if not os.path.exists(processing_folder + ps):
        cmds = 'cp ' + ps + ' ' + processing_folder
        subprocess.check_output(cmds,shell=True)

    if not os.path.exists(processing_folder + stats):
        cmds = 'cp ' + stats + ' ' + processing_folder
        subprocess.check_output(cmds,shell=True)

    if not os.path.exists(processing_folder + rfi):
        cmds = 'cp ' + rfi + ' ' + processing_folder
        subprocess.check_output(cmds,shell=True)

    if not os.path.exists(processing_folder + zaplist):
        cmds = 'cp ' + zaplist + ' ' + processing_folder
        subprocess.check_output(cmds,shell=True)


    current_filterbank2 = processing_folder + os.path.basename(current_filterbank)
    current_filterbank = os.path.basename(current_filterbank2)
  #  current_filterbank3 = processing_folder + os.path.basename(current_filterbank1)
   # current_filterbank1 = os.path.basename(current_filterbank3)
    mask_file1 = processing_folder + os.path.basename(mask_file)
    mask_file = os.path.basename(mask_file1)
    zapfile1= processing_folder + os.path.basename(zaplist)
    zapfile = os.path.basename(zapfile1)



    number_samples1 = 33554432 #2**22 downsampling original file by 4, 2^24 samples in original file
 #       dedisperse_batch1(current_filterbank, number_samples, mask_file)
    val= Parallel(n_jobs = 24)(delayed(dedisperse_batch1)(current_filterbank, number_samples1, mask_file, dm) for dm in np.arange(0.0, 28.252, 0.0552))


   # idedisperse_batch == 2: #down by '2'
    number_samples2 = 16777216 #2**21 
      #  dedisperse_batch2(current_filterbank, number_samples, mask_file)
    val = Parallel(n_jobs = 24)(delayed(dedisperse_batch2)(current_filterbank, number_samples2, mask_file, dm) for dm in np.arange(28.252, 56.5042, 0.11036))

  #  if dedisperse_batch == 3: #down by '4'
    number_samples3 = 8388608 #2**20
      #  dedisperse_batch3(current_filterbank, number_samples, mask_file)
    val = Parallel(n_jobs = 24)(delayed(dedisperse_batch3)(current_filterbank, number_samples3, mask_file, dm) for dm in np.arange(56.5042, 113.0085, 0.2207))


   # if dedisperse_batch == 4: #down by '8'
    number_samples4 = 8388608 #2**19 
       # dedisperse_batch4(current_filterbank, number_samples, mask_file)
    val = Parallel(n_jobs = 24)(delayed(dedisperse_batch4)(current_filterbank, number_samples4, mask_file, dm) for dm in np.arange(113.0085, 226.017, 0.4414))


   # if dedisperse_batch == 5: #down by '16'
    number_samples5 = 8388608 #2**18 
      #  dedisperse_batch5(current_filterbank, number_samples, mask_file)
    val = Parallel(n_jobs = 24)(delayed(dedisperse_batch5)(current_filterbank, number_samples5, mask_file, dm) for dm in np.arange(226.017, 452.034, 0.8828))


   # if dedisperse_batch == 6: #down by '32'
    number_samples6 = 4194304 #2**17
      #  dedisperse_batch6(current_filterbank, number_samples, mask_file)
    val = Parallel(n_jobs = 24)(delayed(dedisperse_batch6)(current_filterbank, number_samples6, mask_file, dm) for dm in np.arange(452.034, 904.068, 1.766))


   # if dedisperse_batch == 7: #down by '32'
    number_samples7 = 2097152 #2**17
      #  dedisperse_batch6(current_filterbank, number_samples, mask_file)
    val = Parallel(n_jobs = 24)(delayed(dedisperse_batch7)(current_filterbank, number_samples7, mask_file, dm) for dm in np.arange(904.068, 1808.136, 3.53))

 #   if dedisperse_batch == 8: #down by '32'
    number_samples8 = 1048576 #2**17
      #  dedisperse_batch6(current_filterbank, number_samples, mask_file)
    val = Parallel(n_jobs = 24)(delayed(dedisperse_batch8)(current_filterbank, number_samples8, mask_file, dm) for dm in np.arange(1808.136, 3002, 7.063))

# is assuming the filterbank files has 16.7 mins of integration time with 54.6 us sampling time. Since restricted by compute power, the analysis starts only by downsampling the original filterbank by a factor of 4 in time, hence the first batch itself has 2^24/4  (for 16.7 mins) number of samples. 
                                                                                                   #
                         
 #   if dedisperse_batch == 1: #down by '1'
    #cmds = 'cp ' + processing_folder + '*.dat *.inf ' + results_location3
#    subprocess.check_output(cmds,shell=True)
 #   cmds = 'rm -rf ' + processing_folder
  #  subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
    t30 = time.time()
    t31 = (t30-t3)
    print 'Time taken for the dedispersion part to execute was %s seconds' %(t31)


############################################################### ACCELSEARCHING THE UNSEGMENTED DAT FILES AND COPYING BACK THE ACCEL FILES ################################################

    t4 = time.time()
   # current_timeseries = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing_name +'/Timeseries/' +  '%s/' %str(beam_number) + 'Unsegmented/'#+ pointing_name + '_%s_' %str(beam_number) + '2power_DM_'
   # os.chdir(current_timeseries)
    cmds = 'ls *dat > dis.txt' #writing all the 1a dedisp dats names to a text file
    subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
    results_location = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/ACCEL_CANDS/'
    mkdir_p(results_location)
    results_location2= '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/ACCEL_CANDS/' + beam
    mkdir_p(results_location2)
    results_location3 = results_location2 + '/' + 'Unsegmented'
    mkdir_p(results_location3)
   # results_location4 = results_location3 + '/' + 'part%s' %str(timeseries_segment)
   # mkdir_p(results_location4)
  #  cmds = 'cp ' + current_timeseries + 'dis.txt ' + processing_folder
   # subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
#copying timeseries to processing folder

   # cmds = 'cp ' + current_timeseries + '*dat ' + processing_folder
   # subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
#copying header to processing folder
   # cmds = 'cp ' + current_timeseries + '*inf ' + processing_folder
   # subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)

    with open('dis.txt') as f:
        data = list(map(str, f))   #storing the dta file names in an array

    s = []
    n = len(data)
    i=0
    for i in range(0, n, 1):
        m = data[i]
        k = m.replace('\n', '')
        s.append(k)                   #correcting dat files name to be used for accelsearch

    


   #time1= processing_folder + pointing_name + '_%s_' %str(beam_number) + '2power_DM_'

   # if accel_batch=='a':
    val = Parallel(n_jobs = 24)(delayed(accelsearch)(s[i], zapfile) for i in range(0, n, 1))


############################################################ doing ACCEL_sifting on the cands =======================================================================================


    root_name = pointing + '_%s_' %str(beam)
  
    #to correct the header names ===========================================================================================================

    cmds = 'ls *DM*inf > inf.txt' #writing all the inf names to a text file
    subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)

    with open('inf.txt') as f:
        data = list(map(str, f))   #storing the inf file names in an array

    s = []
    n = len(data)
    i=0
    for i in range(0, n, 1):
        m = data[i]
        k = m.replace('\n', '')
        s.append(k)

    val = Parallel(n_jobs = 24)(delayed(correct_header)(s[i], root_name) for i in range(0, n, 1))


#to correct the ACCEL search filenames ===========================================================================================================

    cmds = 'ls *ACCEL*100 > ACCEL.txt' #writing all the inf names to a text file
    subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)

    with open('ACCEL.txt') as f1:
        data1 = list(map(str, f1))   #storing the inf file names in an array

    s1 = []
    n1 = len(data1)
    i=0
    for i in range(0, n1, 1):
        m = data1[i]
        k = m.replace('\n', '')
        s1.append(k)

    val = Parallel(n_jobs = 24)(delayed(correct_ACC)(s1[i], root_name) for i in range(0, n1, 1))






#==================================================== SIFTING --------------------------------------------------------------



    txt_file = "SIFTED_CANDIDATES.txt"
    cmds= 'cp ' + '*ACCEL* *inf ' + results_location3
    subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
    
    os.system("python /home/psr/ssengupt/SHALINI_HTRU_North_lowlat/Scripts/one_beam_one_node/sift_Segby0.py %s %s > %s" %(pointing, beam, txt_file))
    
    cmds = "cp " + txt_file + " " + results_location3 
    log = subprocess.check_output(cmds,shell=True)
    print log




    cmds = 'rm -rf *dat' 
    subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
    t40 = time.time()
    t41 = (t40-t4)
    print 'Time taken for the accelsearch and sift part to execute was %s seconds' %(total_n)




######################################################### FOLDING THE UNSEGMENTED =================================================================================================================





    t5 = time.time()
  # folding_batch = int(sys.argv[3])
    


    current_filterbank1 = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing +'/filterbank_files/' +  'beam%s/' %str(beam) + pointing + '_%s_' %str(beam) + '2power.fil' 
    root_filterbank1 = os.path.basename(current_filterbank1[:-4])
    mask_file1 = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/'+ pointing + '/rfi_masks' + '/' + root_filterbank1 + '_rfifind.mask'
   # results_location = '/home/psr/full_kepler_search/cpu_pipeline/TIMESERIES/' + pointing_name + '/'
    bytemask1 = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/'+ pointing + '/rfi_masks' + '/' + root_filterbank1 + '_rfifind.bytemask'
    ps1 = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/'+ pointing + '/rfi_masks' + '/' + root_filterbank1 + '_rfifind.ps'
    stats1= '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/'+ pointing + '/rfi_masks' + '/' + root_filterbank1 + '_rfifind.stats'
    rfi1 = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/'+ pointing + '/rfi_masks' + '/' + root_filterbank1 + '_rfifind.rfi'
    results_location = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/Cand_plots_pipeline/'
    mkdir_p(results_location)
    results_location2= '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/Cand_plots_pipeline/' + beam
    mkdir_p(results_location2)
    results_location3 = results_location2 + '/' + 'Unsegmented_prepdata'
    mkdir_p(results_location3)
 #   results_location4 = results_location3 + '/' + 'part%s' %str(timeseries_segment)
  #  mkdir_p(results_location4)
   # current_search_folder = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing +'/ACCEL_CANDS/' +  '%s/' %str(beam) + 'Unsegmented'
#    processing_folder = '/tmp/data_shalini/'
#    cmds = 'rm -rf ' + processing_folder
 #   log = subprocess.check_output(cmds,shell=True)
  #  print log
   # mkdir_p(processing_folder)
    
   # os.chdir(processing_folder)
    
    if not os.path.exists(processing_folder + current_filterbank1):
        cmds = 'cp ' + current_filterbank1 + ' ' + processing_folder
        subprocess.check_output(cmds,shell=True)

    if not os.path.exists(processing_folder + mask_file1):
        cmds = 'cp ' + mask_file1 + ' ' + processing_folder
        subprocess.check_output(cmds,shell=True)

    if not os.path.exists(processing_folder + bytemask1):
        cmds = 'cp ' + bytemask1 + ' ' + processing_folder
        subprocess.check_output(cmds,shell=True)

    if not os.path.exists(processing_folder + ps1):
        cmds = 'cp ' + ps1 + ' ' + processing_folder
        subprocess.check_output(cmds,shell=True)

    if not os.path.exists(processing_folder + stats1):
        cmds = 'cp ' + stats1 + ' ' + processing_folder
        subprocess.check_output(cmds,shell=True)

    if not os.path.exists(processing_folder + rfi1):
        cmds = 'cp ' + rfi1 + ' ' + processing_folder
        subprocess.check_output(cmds,shell=True)

  #  cmds =  'cp ' + current_search_folder + '/* ' + processing_folder
  #  subprocess.check_output(cmds,shell=True)    

    current_filterbank2 = processing_folder + os.path.basename(current_filterbank1)
    current_filterbank1 = os.path.basename(current_filterbank2)
    mask_file2 = processing_folder + os.path.basename(mask_file1)
    mask_file1 = os.path.basename(mask_file2)


    s=pointing[0]
    print s
    a=[]
    b=[]
    with open('SIFTED_CANDIDATES.txt','r') as f:
        for line in f:
            if line.startswith(s):
                line.split(':')
                a1=line.split(':')[0]
                a2=line.split(':')[1]
                a3=a2.split(' ')[0]
                a.append(a1)
                b.append(a3)
    
    print b
    l1 = len(b)
 #for the folding batches
    l= len(a)
    print l, l1
    k= int((l/6))
    print k, 'divide'
  #  print folding_batch 
   # print a[0], b[0], current_filterbank, mask_file, results_location3



    val = Parallel(n_jobs = 24)(delayed(fold1)(a[i], b[i], current_filterbank1, mask_file1, results_location3, processing_folder) for i in range(0, l, 1))
      #  fold1(a[0], b[0], current_filterbank, mask_file, results_location4)



    cmds = 'cp *bestprof *pfd *ps *png ' + results_location3
    log=subprocess.check_output(cmds,shell=True)
    print log 
    cmds = 'rm -rf ' + processing_folder
    log=subprocess.check_output(cmds,shell=True)
    print log

    t50 = time.time()
    t51 = (t50-t5)
    print 'Time taken for the FOLDING PART to execute was %s seconds' %(t51)

    print 'Time taken for the DEDISPERSING PART to execute was %s seconds' %(t31)

    print 'Time taken for the ACCEL SEARCHING AND SIFTING PART to execute was %s seconds' %(t41)

     #==================================================== STARTING THE SEGMENTED BY 2 PART  =============================================





# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ DEDISPERSING SEQUENTIALLY THE SEG  BY 2  PARTS $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$





#giving d1a d1b d2 to tag it to be used for parallelisation for accelsearch


    t23=time.time()
    for timeseries_segment in range(0, 2, 1):
        os.chdir('/tmp')
        processing_folder1 = 'segment%s' %str(timeseries_segment) + '/'
        cmds = 'rm -rf ' + processing_folder1
        subprocess.check_output(cmds,shell=True)
        mkdir_p(processing_folder1)
#        mkdir_p('hi')
        os.chdir(processing_folder1)
        processing_folder = '/tmp/' + 'segment%s' %str(timeseries_segment) + '/'
        current_filterbank = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing +'/filterbank_files/' +  'beam%s/' %str(beam) + pointing + '_beam_' + str(beam) + '_segment_By_2'  + '_part_' + str(timeseries_segment) + '.fil'
        root_filterbank = os.path.basename(current_filterbank[:-4]) 
        mask_file = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/'+ pointing + '/rfi_masks' + '/' + root_filterbank + '_rfifind.mask'
   # results_location = '/home/psr/full_kepler_search/cpu_pipeline/TIMESERIES/' + pointing_name + '/'
        bytemask = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/'+ pointing + '/rfi_masks' + '/' + root_filterbank + '_rfifind.bytemask'
        ps = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/'+ pointing + '/rfi_masks' + '/' + root_filterbank + '_rfifind.ps'
        stats= '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/'+ pointing + '/rfi_masks' + '/' + root_filterbank + '_rfifind.stats'
        rfi = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/'+ pointing + '/rfi_masks' + '/' + root_filterbank + '_rfifind.rfi'
        results_location = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/Timeseries/' 
        mkdir_p(results_location)
        results_location2= '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/Timeseries/' + beam
        mkdir_p(results_location2)
        results_location3 = results_location2 + '/' + 'Segmented_By_2'
        mkdir_p(results_location3)
        results_location4 = results_location3 + '/' + 'part%s' %str(timeseries_segment)
        mkdir_p(results_location4)
        '''cmds = 'rm -rf ' + processing_folder
        subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
        mkdir_p(processing_folder)
        os.chdir(processing_folder)'''

        ''' Copy filterbank and rfi mask to processing directory and then run the code '''

        if not os.path.exists(processing_folder + current_filterbank):
            cmds = 'cp ' + current_filterbank + ' ' + processing_folder
            subprocess.check_output(cmds,shell=True)

        if not os.path.exists(processing_folder + mask_file):
            cmds = 'cp ' + mask_file + ' ' + processing_folder
            subprocess.check_output(cmds,shell=True)

        if not os.path.exists(processing_folder + bytemask):
            cmds = 'cp ' + bytemask + ' ' + processing_folder
            subprocess.check_output(cmds,shell=True)

        if not os.path.exists(processing_folder + ps):
            cmds = 'cp ' + ps + ' ' + processing_folder
            subprocess.check_output(cmds,shell=True)

        if not os.path.exists(processing_folder + stats):
            cmds = 'cp ' + stats + ' ' + processing_folder
            subprocess.check_output(cmds,shell=True)

        if not os.path.exists(processing_folder + rfi):
            cmds = 'cp ' + rfi + ' ' + processing_folder
            subprocess.check_output(cmds,shell=True)

        if not os.path.exists(processing_folder + zaplist):
            cmds = 'cp ' + zaplist + ' ' + processing_folder
            subprocess.check_output(cmds,shell=True)
       
        current_filterbank1 = processing_folder + os.path.basename(current_filterbank)
        current_filterbank = os.path.basename(current_filterbank1)
        mask_file1 = processing_folder + os.path.basename(mask_file)
        mask_file = os.path.basename(mask_file1)
        zapfile1 = processing_folder + os.path.basename(zaplist)
	zapfile = os.path.basename(zapfile1)

#This is assuming the filterbank files has 16.7 mins of integration time with 54.6 us sampling time. Since restricted by compute power, the analysis starts only by downsampling the original filterbank by a factor of 4 in time, hence the first batch itself has 2^24/4  (for 16.7 mins) number of samples. 
      # if dedisperse_batch == 1: #down by '1'
        number_samples1 = int(8388608/2) #2**22 downsampling original file by 4, 2^24 samples in original fil      
       # dedisperse_batch1(current_filterbank, number_samples, mask_file)
        val= Parallel(n_jobs = 24)(delayed(Sdedisperse_batch1)(current_filterbank, number_samples1, mask_file, dm, processing_folder) for dm in np.arange(0.0, 113.0095, 0.2207))
   #     val = Parallel(n_jobs=num_cores)(delayed(split_data_filterbank)(pointing, fraction, beams[beam_number], output_location) for beam_number in range(len(beams)))

#    if dedisperse_batch == 2: #down by '2'
        number_samples2 = int(8388608/2) #2**21 
       # dedisperse_batch2(current_filterbank, number_samples, mask_file)
        val= Parallel(n_jobs = 24)(delayed(Sdedisperse_batch2)(current_filterbank, number_samples2, mask_file, dm, processing_folder) for dm in np.arange(113.0095, 226.017, 0.4414))

 #   if dedisperse_batch == 3: #down by '4'
        number_samples3 = int(8388608/2) #2**20
   #     dedisperse_batch3(current_filterbank, number_samples, mask_file)
        val = Parallel(n_jobs = 24)(delayed(Sdedisperse_batch3)(current_filterbank, number_samples3, mask_file, dm, processing_folder) for dm in np.arange(226.0170, 452.034, 0.8828))

#    if dedisperse_batch == 4: #down by '8'
        number_samples4 = int(4194304/2) #2**19 
#        dedisperse_batch4(current_filterbank, number_samples, mask_file)
        val = Parallel(n_jobs = 24)(delayed(Sdedisperse_batch4)(current_filterbank, number_samples4, mask_file, dm, processing_folder) for dm in np.arange(452.034, 904.068, 1.76576))


   # if dedisperse_batch == 5: #down by '16'
        number_samples5 = int(2097152/2) #2**18 
         # dedisperse_batch5(current_filterbank, number_samples, mask_file)
        val = Parallel(n_jobs = 24)(delayed(Sdedisperse_batch5)(current_filterbank, number_samples5, mask_file, dm, processing_folder) for dm in np.arange(904.068, 1808.136, 3.5313))


#    if dedisperse_batch == 6: #down by '32'
        number_samples6 = int(1048576/2) #2**17
       # dedisperse_batch6(current_filterbank, number_samples, mask_file)
        val = Parallel(n_jobs = 24)(delayed(Sdedisperse_batch6)(current_filterbank, number_samples6, mask_file, dm, processing_folder) for dm in np.arange(1808.136, 3002, 7.063))
                                                                                                                        
            #cmds = 'cp ' + processing_folder + '*.dat *.inf ' + results_location4
        #subprocess.check_output(cmds,shell=True)
        #cmds = 'rm -rf ' + processing_folder
        #subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
        #t1 = time.time()
        #total_n = (t1-t0)
        #print 'Time taken for the code to execute was %s seconds' %(total_n)
   

# ======================================= ACCELSEARCHING THE SEGMENTED BY 2 PART IN A LOOP =============================================================

 

        cmds = 'ls *dat > dis.txt' #writing all the 1a dedisp dats names to a text file
        subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
        results_location = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/ACCEL_CANDS/'
        mkdir_p(results_location)
        results_location2= '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/ACCEL_CANDS/' + beam
        mkdir_p(results_location2)
        results_location3 = results_location2 + '/' + 'Segmented_By_2'
        mkdir_p(results_location3)
        results_location4 = results_location3 + '/' + 'part%s' %str(timeseries_segment)
        mkdir_p(results_location4)
   # results_location4 = results_location3 + '/' + 'part%s' %str(timeseries_segment)
   # mkdir_p(results_location4)
  #  cmds = 'cp ' + current_timeseries + 'dis.txt ' + processing_folder
   # subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
#copying timeseries to processing folder

   # cmds = 'cp ' + current_timeseries + '*dat ' + processing_folder
   # subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
#copying header to processing folder
   # cmds = 'cp ' + current_timeseries + '*inf ' + processing_folder
   # subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
        ta = time.time()
        with open('dis.txt') as f:
            data = list(map(str, f))   #storing the dta file names in an array

        s = []
        n = len(data)
        i=0
        for i in range(0, n, 1):
            m = data[i]
            k = m.replace('\n', '')
            s.append(k)                   #correcting dat files name to be used for accelsearch

    


   #time1= processing_folder + pointing_name + '_%s_' %str(beam_number) + '2power_DM_'

   # if accel_batch=='a':
        val = Parallel(n_jobs = 24)(delayed(accelsearch_S2)(s[i], processing_folder, zapfile) for i in range(0, n, 1))


############################################################ doing ACCEL_sifting on the cands =======================================================================================


        root_name = pointing + '_%s_' %str(beam) + 'Seg_by2_' + 'part_%s' %str(timeseries_segment)

        #to correct the header names ===========================================================================================================

        cmds = 'ls *DM*inf > inf.txt' #writing all the inf names to a text file
        subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)

        with open('inf.txt') as f:
            data = list(map(str, f))   #storing the inf file names in an array

        s = []
        n = len(data)
        i=0
        for i in range(0, n, 1):
            m = data[i]
            k = m.replace('\n', '')
            s.append(k)

        val = Parallel(n_jobs = 24)(delayed(Scorrect_header)(s[i], root_name, processing_folder) for i in range(0, n, 1))


#to correct the ACCEL search filenames ===========================================================================================================

        cmds = 'ls *ACCEL*668 > ACCEL.txt' #writing all the inf names to a text file
        subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)

        with open('ACCEL.txt') as f1:
            data1 = list(map(str, f1))   #storing the inf file names in an array

        s1 = []
        n1 = len(data1)
        i=0
        for i in range(0, n1, 1):
            m = data1[i]
            k = m.replace('\n', '')
            s1.append(k)


        val = Parallel(n_jobs = 24)(delayed(Scorrect_ACC)(s1[i], root_name, processing_folder) for i in range(0, n1, 1))







        txt_file = "SIFTED_CANDIDATES.txt"
        cmds= 'cp ' + '*ACCEL* *inf ' + results_location4
        subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
    
        os.system("python /home/psr/ssengupt/SHALINI_HTRU_North_lowlat/Scripts/one_beam_one_node/sift_Segby2.py %s %s %s > %s" %(pointing, beam, timeseries_segment, txt_file))
    
        cmds = "cp " + txt_file + " " + results_location4 
        log = subprocess.check_output(cmds,shell=True)
        print log




        cmds = 'rm -rf *dat' 
        subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
        ta1 = time.time()
        ta0 = (ta1-ta)
      #  print 'Time taken for the accelsearch and sift part to execute was %s seconds' %(total_n)

# ========================================================== FOLDING THE SEGEMENTED BY 2 CANDS ================================================================================         

        current_filterbank1 = processing_folder + os.path.basename(current_filterbank)
        current_filterbank = os.path.basename(current_filterbank1)
        mask_file2 = processing_folder + os.path.basename(mask_file)
        mask_file = os.path.basename(mask_file2)

        results_location = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/Cand_plots_pipeline/'
        mkdir_p(results_location)
        results_location2= '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/Cand_plots_pipeline/' + beam
        mkdir_p(results_location2)
        results_location3 = results_location2 + '/' + 'Segmented_By_2'
        mkdir_p(results_location3)
        results_location4 = results_location3 + '/' + 'part%s' %str(timeseries_segment)
        mkdir_p(results_location4)
        tf = time.time()
        s=pointing[0]
        print s
        a=[]
        b=[]
        with open('SIFTED_CANDIDATES.txt','r') as f:
            for line in f:
                if line.startswith(s):
                    line.split(':')
                    a1=line.split(':')[0]
                    a2=line.split(':')[1]
                    a3=a2.split(' ')[0]
                    a.append(a1)
                    b.append(a3)
    
        print b
        l1 = len(b)
 #for the folding batches
        l= len(a)
        print l, l1
        k= int((l/6))
        print k, 'divide'
  #  print folding_batch 
   # print a[0], b[0], current_filterbank, mask_file, results_location3



        val = Parallel(n_jobs = 24)(delayed(fold1)(a[i], b[i], current_filterbank, mask_file, results_location4, processing_folder) for i in range(0, l, 1))
      #  fold1(a[0], b[0], current_filterbank, mask_file, results_location4)



        cmds = 'cp *bestprof *pfd *ps *png ' + results_location4
        log=subprocess.check_output(cmds,shell=True)
        print log 
        os.chdir('/tmp')
       # print "should print /tmp now"
        cmds= 'pwd' #should print /tmp
        log = subprocess.check_output(cmds,shell=True)
        print log
        cmds = 'rm -rf ' + processing_folder1
        log=subprocess.check_output(cmds,shell=True)
        print log
        print "should print hi now"
        cmds= 'ls' #should print hi
        log = subprocess.check_output(cmds,shell=True)
        print log

        tf1= time.time()
        tf0 = tf1-tf








# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ DEDISPERSING SEQUENTIALLY THE SEG  BY 4  PARTS $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$





#giving d1a d1b d2 to tag it to be used for parallelisation for accelsearch


    t23=time.time()
    for timeseries_segment in range(0, 4, 1):
        os.chdir('/tmp')
        processing_folder1 = '4segment%s' %str(timeseries_segment) + '/'
        cmds = 'rm -rf ' + processing_folder1
        subprocess.check_output(cmds,shell=True)
        mkdir_p(processing_folder1)
        mkdir_p('hi')
        os.chdir(processing_folder1)
        processing_folder = '/tmp/' + '4segment%s' %str(timeseries_segment) + '/'

#rocessing_folder = '/tmp/' + 'segment%s' %str(timeseries_segment) + '/'
        current_filterbank = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing +'/filterbank_files/' +  'beam%s/' %str(beam) + pointing + '_beam_' + str(beam) + '_segment_By_4'  + '_part_' + str(timeseries_segment) + '.fil'
        root_filterbank = os.path.basename(current_filterbank[:-4]) 
        mask_file = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/'+ pointing + '/rfi_masks' + '/' + root_filterbank + '_rfifind.mask'
   # results_location = '/home/psr/full_kepler_search/cpu_pipeline/TIMESERIES/' + pointing_name + '/'
        bytemask = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/'+ pointing + '/rfi_masks' + '/' + root_filterbank + '_rfifind.bytemask'
        ps = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/'+ pointing + '/rfi_masks' + '/' + root_filterbank + '_rfifind.ps'
        stats= '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/'+ pointing + '/rfi_masks' + '/' + root_filterbank + '_rfifind.stats'
        rfi = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/'+ pointing + '/rfi_masks' + '/' + root_filterbank + '_rfifind.rfi'
        results_location = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/Timeseries/' 
        mkdir_p(results_location)
        results_location2= '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/Timeseries/' + beam
        mkdir_p(results_location2)
        results_location3 = results_location2 + '/' + 'Segmented_By_4_pipe_prepdata'
        mkdir_p(results_location3)
        results_location4 = results_location3 + '/' + 'part%s' %str(timeseries_segment)
        mkdir_p(results_location4)
       # os.chdir(results_location4)

        if not os.path.exists(processing_folder + current_filterbank):
            cmds = 'cp ' + current_filterbank + ' ' + processing_folder
            subprocess.check_output(cmds,shell=True)

        if not os.path.exists(processing_folder + mask_file):
            cmds = 'cp ' + mask_file + ' ' + processing_folder
            subprocess.check_output(cmds,shell=True)

        if not os.path.exists(processing_folder + bytemask):
            cmds = 'cp ' + bytemask + ' ' + processing_folder
            subprocess.check_output(cmds,shell=True)

        if not os.path.exists(processing_folder + ps):
            cmds = 'cp ' + ps + ' ' + processing_folder
            subprocess.check_output(cmds,shell=True)

        if not os.path.exists(processing_folder + stats):
            cmds = 'cp ' + stats + ' ' + processing_folder
            subprocess.check_output(cmds,shell=True)

        if not os.path.exists(processing_folder + rfi):
            cmds = 'cp ' + rfi + ' ' + processing_folder
            subprocess.check_output(cmds,shell=True) 
 
          

        if not os.path.exists(processing_folder + zaplist):
            cmds = 'cp ' + zaplist + ' ' + processing_folder
            subprocess.check_output(cmds,shell=True)
 #    print "should print only the filterbank and masks now"
        cmds= 'ls'    #sgould print filterbank and masks
        log = subprocess.check_output(cmds,shell=True)
        print log
        current_filterbank1 = processing_folder + os.path.basename(current_filterbank)
        current_filterbank = os.path.basename(current_filterbank1)
        mask_file1 = processing_folder + os.path.basename(mask_file)
        mask_file = os.path.basename(mask_file1)
        zapfile1 = processing_folder + os.path.basename(zaplist)
	zapfile = os.path.basename(zapfile1)



#This is assuming the filterbank files has 16.7 mins of integration time with 54.6 us sampling time. Since restricted by compute power, the analysis starts only by downsampling the original filterbank by a factor of 4 in time, hence the first batch itself has 2^24/4  (for 16.7 mins) number of samples. 
        

        number_samples1 = int(8388608/4) #2**22 downsampling original file by 4, 2^24 samples in original file
          #  dedisperse_batch1(current_filterbank, number_samples, mask_file)
        val1= Parallel(n_jobs = 24)(delayed(Sdedisperse_batch1)(current_filterbank, number_samples1, mask_file, dm, processing_folder) for dm in np.arange(0.0, 113.0095, 0.2207))

          #      elif dedisperse_batch == 2: #down by '2'
        number_samples2 = int(8388608/4) #2**21 
         #   dedisperse_batch2(current_filterbank, number_samples, mask_file)
        val2= Parallel(n_jobs = 24)(delayed(Sdedisperse_batch2)(current_filterbank, number_samples2, mask_file, dm, processing_folder) for dm in np.arange(113.0095, 226.017, 0.4414))

              #     elif dedisperse_batch == 3: #down by '4'
        number_samples3 = int(8388608/4) #2**20
          #  dedisperse_batch3(current_filterbank, number_samples, mask_file)
        val3 = Parallel(n_jobs = 24)(delayed(Sdedisperse_batch3)(current_filterbank, number_samples3, mask_file, dm, processing_folder) for dm in np.arange(226.0170, 452.034, 0.8828))

         #        elif dedisperse_batch == 4: #down by '8'
        number_samples4 = int(4194304/4) #2**19 
          #  dedisperse_batch4(current_filterbank, number_samples, mask_file)
        val4 = Parallel(n_jobs = 24)(delayed(Sdedisperse_batch4)(current_filterbank, number_samples4, mask_file, dm, processing_folder) for dm in np.arange(452.034, 904.068, 1.76576))

        #      elif dedisperse_batch == 5: #down by '16'
        number_samples5 = int(2097152/4) #2**18 
         #   dedisperse_batch5(current_filterbank, number_samples, mask_file
        val5 = Parallel(n_jobs = 24)(delayed(Sdedisperse_batch5)(current_filterbank, number_samples5, mask_file, dm, processing_folder) for dm in np.arange(904.068, 1808.136, 3.5313))

   #     else:# dedisperse_batch == 6: #down by '32'
        number_samples6 = int(1048576/4) #2**17
           # dedisperse_batch6(current_filterbank, number_samples, mask_file)
        val6 = Parallel(n_jobs = 24)(delayed(Sdedisperse_batch6)(current_filterbank, number_samples6, mask_file, dm, processing_folder) for dm in np.arange(1808.136, 3002, 7.063))






                                                                                                                    
    #    cmds = 'cp ' + '*.dat *.inf ' + results_location4
     #   subprocess.check_output(cmds,shell=True)
     #   print "should print only the fil and mask files and dat and inf"
        cmds= 'ls'   #should print dat and inf and mask and fil
        log = subprocess.check_output(cmds,shell=True)
        print log
        #cmds = 'rm -rf ' + processing_folder
        #subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
        #t1 = time.time()
        #total_n = (t1-t0)
        #print 'Time taken for the code to execute was %s seconds' %(total_n)
   

# ======================================= ACCELSEARCHING THE SEGMENTED BY 2 PART IN A LOOP =============================================================

 

        cmds = 'ls *dat > dis.txt' #writing all the 1a dedisp dats names to a text file
        subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
        cmds = 'cp dis.txt ' + results_location4
        subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
        results_location = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/ACCEL_CANDS/'
        mkdir_p(results_location)
        results_location2= '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/ACCEL_CANDS/' + beam
        mkdir_p(results_location2)
        results_location3 = results_location2 + '/' + 'Segmented_By_4'
        mkdir_p(results_location3)
        results_location4 = results_location3 + '/' + 'part%s' %str(timeseries_segment)
        mkdir_p(results_location4)
        ta = time.time()
        with open('dis.txt') as f:
            data = list(map(str, f))   #storing the dta file names in an array

        s = []
        n = len(data)
        i=0
        for i in range(0, n, 1):
            m = data[i]
            k = m.replace('\n', '')
            s.append(k)                   #correcting dat files name to be used for accelsearch

    

     
        val = Parallel(n_jobs = 24)(delayed(accelsearch_S4)(s[i], processing_folder, zapfile) for i in range(0, n, 1))


############################################################ doing ACCEL_sifting on the cands =======================================================================================

        root_name = pointing + '_%s_' %str(beam) + 'Seg_by4_' + 'part_%s' %str(timeseries_segment)

         #to correct the header names ===========================================================================================================

        cmds = 'ls *DM*inf > inf.txt' #writing all the inf names to a text file
        subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)

        with open('inf.txt') as f:
            data = list(map(str, f))   #storing the inf file names in an array

        s = []
        n = len(data)
        i =0
        for i in range(0, n, 1):
            m = data[i]
            k = m.replace('\n', '')
            s.append(k)

        val = Parallel(n_jobs = 24)(delayed(S4correct_header)(s[i], root_name, processing_folder) for i in range(0, n, 1))


#to correct the ACCEL search filenames ===========================================================================================================

        cmds = 'ls *ACCEL*422 > ACCEL.txt' #writing all the inf names to a text file
        subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)

        with open('ACCEL.txt') as f1:
            data1 = list(map(str, f1))   #storing the inf file names in an array

        s1 = []
        n1 = len(data1)
        i=0
        for i in range(0, n1, 1):
            m = data1[i]
            k = m.replace('\n', '')
            s1.append(k)

        val = Parallel(n_jobs = 24)(delayed(S4correct_ACC)(s1[i], root_name, processing_folder) for i in range(0, n1, 1))



#===================================== sifting  ====================================

        txt_file = "SIFTED_CANDIDATES.txt"
        cmds= 'cp ' + '*ACCEL* *inf ' + results_location4
        subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
        
      #  print "will print current /tmp/segment x directory now"
        cmds= 'pwd'
        log = subprocess.check_output(cmds,shell=True)
        print log
       # print "will print all the files now, dat accle, fil mask"
        cmds= 'ls'   #should print dat and accel
        log = subprocess.check_output(cmds,shell=True)
        print log
        os.system("python /home/psr/ssengupt/SHALINI_HTRU_North_lowlat/Scripts/one_beam_one_node/sift_Segby4.py %s %s %s > %s" %(pointing, beam, timeseries_segment, txt_file))
    
        cmds = "cp " + txt_file + " " + results_location4 
        log = subprocess.check_output(cmds,shell=True)
        print log




        cmds = 'rm -rf *dat' 
        subprocess.check_output(cmds,shell=True, stderr=subprocess.STDOUT)
        ta1 = time.time()
        ta0 = (ta1-ta)
      #  print 'Time taken for the accelsearch and sift part to execute was %s seconds' %(total_n)

# ========================================================== FOLDING THE SEGEMENTED BY 2 CANDS ================================================================================         

        current_filterbank1 = processing_folder + os.path.basename(current_filterbank)
        current_filterbank = os.path.basename(current_filterbank1)
        mask_file2 = processing_folder + os.path.basename(mask_file)
        mask_file = os.path.basename(mask_file2)

        results_location = '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/Cand_plots_pipeline/'
        mkdir_p(results_location)
        results_location2= '/home/psr/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/' + pointing + '/Cand_plots_pipeline/' + beam
        mkdir_p(results_location2)
        results_location3 = results_location2 + '/' + 'Segmented_By_4'
        mkdir_p(results_location3)
        results_location4 = results_location3 + '/' + 'part%s' %str(timeseries_segment)
        mkdir_p(results_location4)
        tf = time.time()
        s=pointing[0]
        print s
        a=[]
        b=[]
        with open('SIFTED_CANDIDATES.txt','r') as f:
            for line in f:
                if line.startswith(s):
                    line.split(':')
                    a1=line.split(':')[0]
                    a2=line.split(':')[1]
                    a3=a2.split(' ')[0]
                    a.append(a1)
                    b.append(a3)
    
        print b
        l1 = len(b)
 #for the folding batches
        l= len(a)
        print l, l1
        k= int((l/6))
        print k, 'divide'
  #  print folding_batch 
   # print a[0], b[0], current_filterbank, mask_file, results_location3



        val = Parallel(n_jobs = 24)(delayed(fold1)(a[i], b[i], current_filterbank, mask_file, results_location4, processing_folder) for i in range(0, l, 1))
      #  fold1(a[0], b[0], current_filterbank, mask_file, results_location4)

        cmds = 'cp *bestprof *pfd *ps *png ' + results_location4
        log=subprocess.check_output(cmds,shell=True)
        print log 
        os.chdir('/tmp')
       # print "should print /tmp now"
        cmds= 'pwd' #should print /tmp
        log = subprocess.check_output(cmds,shell=True)
        print log
        cmds = 'rm -rf ' + processing_folder1
        log=subprocess.check_output(cmds,shell=True)
        print log
        print "should print hi now"
        cmds= 'ls' #should print hi
        log = subprocess.check_output(cmds,shell=True)
        print log
        tf1= time.time()
        os.chdir(results_location4)
        tf0 = tf1-tf







    tfull0=time.time()
    t = tfull0 - tfull
    tp0= time.time()
    #tp1=tp0-tp
    print 'time taken for both the segmented full pipeline sequentially to run in seconds %s ' %(t) 
    
