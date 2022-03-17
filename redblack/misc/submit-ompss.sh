#!/bin/csh
# following option makes sure the job will run in the current directory
#$ -cwd
# following option makes sure the job has the same environmnent variables as the submission shell
#$ -V


setenv PROG heat-ompss

echo "Sequential execution"
./heat testgrind2.dat

set n_threads = 1
set MAX_THREADS = 12
while ($n_threads <= $MAX_THREADS)
    echo "OmpSs execution with $n_threads threads"
    setenv OMP_NUM_THREADS $n_threads
    ./$PROG testgrind2.dat
    @ n_threads = $n_threads * 2
end
