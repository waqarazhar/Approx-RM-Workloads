#include "heat-ompss.h"
#include <stdlib.h>

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )


/*
 * Blocked Red-Black solver: one iteration step
 */
void relax_redblack (unsigned padding, unsigned sizey, double (*u)[sizey+padding*2], unsigned sizex, int check, double *residual  )
{
    int nbx, bx, nby, by;
    int inf_i, sup_i, inf_j, sup_j;
    int lsw;

    nbx = 2;
    bx = sizex/nbx;
    nby = 2;
    by = sizey/nby;
    
    double *sum;
    double (*local_sum)[nby];
    posix_memalign( (void**)&local_sum, sizeof( double) * nbx *nby, sizeof( double) * nbx *nby );
    
    if ( check ) 
    {
        posix_memalign( (void**)&sum, sizeof( double ), sizeof( double ) );
        *sum=0.0;
    }

    
    // Computing "Red" blocks
    for (int ii=0; ii<nbx; ii++) 
    {
        lsw = ii%2;
        for (int jj=lsw; jj<nby; jj=jj+2) 
        {
            inf_i = padding+1+ii*bx;
            inf_j = padding+1+jj*by;
            sup_i = min(padding+1+(ii+1)*bx, padding+sizex-1);
            sup_j = min(padding+1+(jj+1)*by, padding+sizey-1);

            #pragma omp task 

            {
                if( check ) local_sum[ii][jj] = 0.0f;

                for (int i=inf_i; i < sup_i; i++) 
                    for (int j=inf_j; j < sup_j; j++) 
                    {
                        double unew, diff;
                        unew= 0.25 * (    u[ i][ (j-1) ]+  // left
                                          u[ i][(j+1) ]+  // right
                                          u[ (i-1)][ j ]+  // top
                                          u[ (i+1)][ j ]); // bottom
                        if ( check ) {
                            diff = unew - u[i][j];
                            local_sum[ii][jj] += diff * diff;
                        }
                        u[i][j]=unew;
                    }
            }
        }
    }

     // Sync all task before computing the sum
    # pragma omp taskwait

    // ########################################
    // Compute the sum
    # pragma omp task 
    {
        for (int ii=0; ii<nbx; ii++) 
        {
            lsw = ii%2;
            for (int jj=lsw; jj<nby; jj=jj+2) 
            {
                    *sum += local_sum [ii][jj];
            }
    }



    // ################################


    // Computing "Black" blocks
    for (int ii=0; ii<nbx; ii++) {
        lsw = (ii+1)%2;
        for (int jj=lsw; jj<nby; jj=jj+2) {
            inf_i = padding+1+ii*bx;
            inf_j = padding+1+jj*by;
            sup_i = min(padding+1+(ii+1)*bx, padding+sizex-1);
            sup_j = min(padding+1+(jj+1)*by, padding+sizey-1);

           
            #pragma omp task 
            {
                if( check ) local_sum[ii][jj] = 0.0f;
                for (int i=inf_i; i < sup_i; i++) 
                    for (int j=inf_j; j < sup_j; j++) 
                    {
                        double unew, diff;
                        unew= 0.25 * (    u[ i][ (j-1) ]+  // left
                                          u[ i][ (j+1) ]+  // right
                                          u[ (i-1)][ j     ]+  // top
                                          u[ (i+1)][ j     ]); // bottom
                        if ( check ) {
                            diff = unew - u[i][j];
                            local_sum[ii][jj] += diff * diff;
                        }
                        u[i][j]=unew;
                    }
            }
        }
    }

    // Sync all task before computing the sum

        # pragma omp taskwait
       
        #pragma omp task 
        {
            for (int ii=0; ii<nbx; ii++) {
                lsw = (ii+1)%2;
                for (int jj=lsw; jj<nby; jj=jj+2)
                    {
                        *sum += local_sum [ii][jj];
                    }
                }

            *residual = *sum;
            // Release local variables
            free( local_sum );
            free( sum );
        }
    }


}

