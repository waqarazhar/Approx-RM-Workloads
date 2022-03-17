#!/bin/csh
# following option makes sure the job will run in the current directory
#$ -cwd
# following option makes sure the job has the same environmnent variables as the submission shell
#$ -V

setenv PROG heat-ompss-i
setenv OMP_NUM_THREADS 8

setenv  NX_INSTRUMENTATION extrae


./$PROG testgrind2.dat
