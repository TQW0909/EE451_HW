#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int main(int argc, char *argv[]){

        if (argc < 2) {
            printf("Missing block size!!!!");
            return 0;
        }
        int b = argv[1];
        if (b % 4 != 0)
        {
            printf("Block size needs to be mutiple of 4!!!");
            return 0;
        }
		int i, j, k;
		struct timespec start, stop; 
		double time;
		int n = 4096; // matrix size is n*n
		
		double **A = (double**) malloc (sizeof(double*)*n/b);
		double **B = (double**) malloc (sizeof(double*)*n/b);
		double **C = (double**) malloc (sizeof(double*)*n/b);

		for (i=0; i<n/b; i++) {
			A[i] = (double*) malloc(sizeof(double)*n/b);
			B[i] = (double*) malloc(sizeof(double)*n/b);
			C[i] = (double*) malloc(sizeof(double)*n/b);
		}

        // Below has not been modified
		
		for (i=0; i<n; i++){
			for(j=0; j< n; j++){
				A[i][j]=i;
				B[i][j]=i+j;
				C[i][j]=0;			
			}
		}
				
		if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
		
		// Your code goes here //
		// Matrix C = Matrix A * Matrix B //	
		//*******************************//
		
		
		//*******************************//
		
		if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
		time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
		
		printf("Number of FLOPs = %lu, Execution time = %f sec,\n%lf MFLOPs per sec\n", 2*n*n*n, time, 1/time/1e6*2*n*n*n);		
		printf("C[100][100]=%f\n", C[100][100]);
		
		// release memory
		for (i=0; i<n; i++) {
			free(A[i]);
			free(B[i]);
			free(C[i]);
		}
		free(A);
		free(B);
		free(C);
		return 0;
}
