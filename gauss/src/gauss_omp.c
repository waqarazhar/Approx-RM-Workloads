#include "heat-ompss.h"
#include <stdlib.h>

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )

/*
 * Blocked Gauss-Seidel solver: one iteration step
 */
void relax_gauss (unsigned padding, unsigned sizey, double (*u)[sizey+padding*2], unsigned sizex, int check, double *residual )
{
    // gmiranda: Do not use variables in the stack for tasks since they might be destroyed
    double *sum;

    int nbx, bx, nby, by;
    int inf_i, sup_i, inf_j, sup_j;

    nbx = 2;
    bx = sizex/nbx;
    nby = 2;
    by = sizey/nby;
    
    //extern int Task_Cntr_Out;

    // gmiranda: same, stack vars are not allowed!
    double (*local_sum)[nby];
    posix_memalign( (void**)&local_sum, sizeof( double) * nbx *nby, sizeof( double) * nbx *nby );

    if ( check ) {
        posix_memalign( (void**)&sum, sizeof( double ), sizeof( double ) );
        //#pragma omp task out (*sum) label(init_sum)
        *sum=0.0;
    }
   	
    
   //	#pragma omp parallel num_threads(4) 
  	//{
  	//#pragma omp master //
  	//{	
    	for (int ii=padding+1; ii<sizex-1+padding; ii+=bx)
    	{
        	for (int jj=padding+1; jj<sizey-1+padding; jj+=by) 
        	{
				inf_i = ii; inf_j=jj; 
	 			sup_i = (ii+bx)<sizex-1 + padding ? ii+bx : sizex-1 + padding; 	
	 			sup_j = (jj+bx)<sizey-1 + padding ? jj+by : sizey-1 + padding;
				// north, south, west, east
				// Orgonal ompss implementation has array section syntax that is not compatible with openMP.
		
	           //fprintf(stdout,"omp_get_thread_num %u \n",omp_get_thread_num());   

				#pragma omp task 
				{
    //                           fprintf(stdout,"omp_get_thread_num T1 %u \n",omp_get_thread_num());            
            	   if ( check )
                    local_sum[(ii-padding)/bx][(jj-padding)/by]=0.0f;
	
            	   for (int i=inf_i; i<sup_i; i++)
                	   for (int j=inf_j; j<sup_j; j++) 
                	   {
                    	   double unew, diff;
                    	   unew= 0.25 * (    u[i][(j-1)] + u[i][(j+1)] + u[(i-1)][j] + u[(i+1)][j]); 
                    	   if ( check ) 
                    	   {
                        	   diff = unew - u[i][j];
                        	   local_sum [(ii-padding)/bx][(jj-padding)/by] += diff * diff;
                    	   }
                    	   u[i][j]=unew;
                	   }
            	}
            
        		//if ( check ) 
        		//{
        			//#pragma omp taskwait
                   
            	//	#pragma omp task 
                 //   {	
                		//*sum += local_sum [ (ii-padding)/bx ][ (jj-padding)/by ];
            	//	}
        		//}

        	}
    	}

    #pragma omp taskwait
    #pragma omp task 
    {    
     for (int ii=padding+1; ii<sizex-1+padding; ii+=bx)
        {
            for (int jj=padding+1; jj<sizey-1+padding; jj+=by) 
            {
                *sum += local_sum [ (ii-padding)/bx ][ (jj-padding)/by ];
            }
        }
        *residual = *sum;
        // Release local variables
        free( local_sum );
        free( sum );    
    }

    //if ( check ) 
    //{
    //    #pragma omp taskwait
    //    #pragma omp task 
    //    {
    //    	//printf(" Sum parallel %f \n",sum);
            
    //    }
    //}
    //else 
    //{
        // Free local_sum as it will not be released if check is false
      //  free( local_sum );
    //}

	//} // end Omp single
	//} // end omp parallel
}
