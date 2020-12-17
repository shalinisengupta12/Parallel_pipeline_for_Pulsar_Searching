#!/bin/bash
pointing=$1
#beam=$2
mkdir -p /u/ssengupt/SHALINI_HTRU_North_lowlat/data_files/Test_poitings/${pointing}
mkdir -p /u/ssengupt/SHALINI_HTRU_North_lowlat/job_logs/${pointing}
sbatch --mail-user=link2shalini@gmail.com -J RFD_${pointing} -o /u/ssengupt/SHALINI_HTRU_North_lowlat/job_logs/${pointing}/RFD_${pointing}.out -e /u/ssengupt/SHALINI_HTRU_North_lowlat/job_logs/${pointing}/RFD_${pointing}.err -t 05:00:0 --partition=long.q --nodes=1 --cpus-per-task=24 --mem=62000 --wrap="singularity exec -H $HOME:/home1 -B /u/ssengupt:/home/psr/ssengupt ~/template_bank_cpu_pipeline.simg python /home/psr/ssengupt/SHALINI_HTRU_North_lowlat/Scripts/one_beam_one_node/RFI_mit_fourier.py ${pointing}"


#export JOBS=`squeue -u ssengupt -n ${beam}prepdata_${pointing} | wc -l` ## MIGREV
#export QUEUE=`squeue -u ssengupt -n ${beam}prepdata_${pointing} | grep 'PD' | wc -l`
#while (( $JOBS > 1 || $QUEUE > 1 ))
#do
#    echo ${beam}prepdata_${pointing}: still queuing $QUEUE , sleep 5s
#    sleep 30
#    export JOBS=`squeue -u ssengupt -n ${beam}prepdata_${pointing} | wc -l` ## MIGREV
#    export QUEUE=`squeue -u ssengupt -n ${beam}prepdata_${pointing} | grep 'PD' | wc -l`
#done

