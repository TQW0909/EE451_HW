#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <pthread.h>
#include <stdint.h>

#define n 4096

double **A, **B, **C;
pthread_mutex_t *mutexes;
int p;

struct  thread_data{
   int	thread_id;
   int	row;
   int	col;
   int 	length;
};

void *compute(void *data){

	struct thread_data * matrix_data;
	matrix_data = (struct thread_data *) data;

	int row = matrix_data->row;
	int col = matrix_data->col;


	for (int i = row; i < row + matrix_data->length; i++)
	{
		for(int j = col; j < col + matrix_data->length; j++)
		{
			for (int k = 0; k < n; k++) 
			{
                pthread_mutex_lock(&mutexes[i / (n/p)]);
				C[i][k] += A[i][j] * B[j][k];
                pthread_mutex_unlock(&mutexes[i / (n/p)]);
			}
		}
	}

	pthread_exit(NULL);
}


int main(int argc, char *argv[]){
		
	if (argc < 2)
	{
		printf("Need to enter number of threads!");
		return -1;
	}

	p = atoi(argv[1]);
	int num_threads = p * p;

	pthread_t threads[num_threads];
	struct thread_data  thread_data_array[num_threads];
    mutexes = (pthread_mutex_t*) malloc (sizeof(pthread_mutex_t)*p);

	int i, j, k, rc;
	struct timespec start, stop; 
	double time;
	// int n = 4096; // matrix size is n*n (Default is 4096)

    for (i = 0; i < p; i++)
    {
        pthread_mutex_init(&mutexes[i],NULL);
    }
	
	A = (double**) malloc (sizeof(double*)*n);
	B = (double**) malloc (sizeof(double*)*n);
	C = (double**) malloc (sizeof(double*)*n);
	
	for (i=0; i<n; i++) {
		A[i] = (double*) malloc(sizeof(double)*n);
		B[i] = (double*) malloc(sizeof(double)*n);
		C[i] = (double*) malloc(sizeof(double)*n);
	}
	
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

	for (i = 0; i < p; i++)
	{
		for (j = 0; j < p; j++)
		{
			int index = j + i * p;
            thread_data_array[index].thread_id = index;
			thread_data_array[index].col = i * n / p;
			thread_data_array[index].row = j * n / p;
			thread_data_array[index].length = n / p;
			rc = pthread_create(&threads[index], NULL, compute, (void *) &thread_data_array[index]);
			if (rc) { printf("ERROR; return code from pthread_create() is %d\n", rc); exit(-1);}
		}
	}

	for (i = 0; i < num_threads; i++)
	{
		pthread_join(threads[i], NULL);
	}
	
	//*******************************//
	
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
	time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	
	printf("Number of FLOPs = %lld, Execution time = %f sec,\n%lf MFLOPs per sec\n",  (long long)2*n*n*n, time, 1/time/1e6*2*n*n*n);		
	printf("C[100][100]=%f\n", C[100][100]);
	
    for (i = 0; i < p; i++)
    {
        pthread_mutex_destroy(&mutexes[i]);
    }

    free(mutexes);

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
