# Embarassingly_parallel_pipelien_to_find_pulsars_from_Effelsberg_radio_telescope_data
To process terabytes of data coming from 7 beams of radio telescope to sift candidates in the fourier domain to find repeating pulses using the software PRESTO (https://github.com/scottransom/presto), this pipeline has been built with a python base to parallelise the processing of data over various cores of a node, further multiple nodes are used for this purpose in turn. I have used Slurm workload manager to assign nodes of the cluster for my created pipelines. 
