CC     =  g++ 

CFLAGS = -Wall -O2 

LIBS   = -lm 
myperf_dir=../../myperf

ALL	= GD

all: $(ALL)


GD: GD.cpp 
	$(CC) $(CFLAGS) -fopenmp $+ $(LIBS) -o $@


clean:
	rm -fr $(ALL) *.o *.xdot *.out *.err *.prv *.pcf *.row TRACE.mpits set-0