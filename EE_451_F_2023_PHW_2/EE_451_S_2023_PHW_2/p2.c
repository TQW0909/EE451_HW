#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

#define h  800 
#define w  800
#define K  6

#define input_file  "input.raw"
#define output_file "output.raw"

struct  thread_data{
   int	thread_id;
   int	size;
   int  start;
   int  sums[K];
   int  count[K];
}; 

unsigned char *a;
double u[K] = {0, 65, 100, 125, 190, 255}; // Initial cluster mean values


void *compute(void *data){

	struct thread_data * point_data;
	point_data = (struct thread_data *) data;

    for (int i = 0; i < K; i++)
    {
        point_data->count[i] = 0;
        point_data->sums[i] = 0;
    }

    // Finding the clostest cluster to a point
    
    for (int x = point_data->start; x < point_data->start + point_data->size; x++)
    {
        int min = fabs(a[x] - u[0]);
        unsigned char closest = 0;
        for (int y = 1; y < K; y++)
        {
            unsigned char temp = fabs(a[x] - u[y]);
            if (temp < min) 
            {
                min =  temp;
                closest = y;
            }
        }

        point_data->count[closest]++;
        point_data->sums[closest] += a[x];
    }

	pthread_exit(NULL);
}

int main(int argc, char *argv[]){
		
	if (argc < 2)
	{
		printf("Need to enter number of threads!\n");
		return -1;
	}

	int p = atoi(argv[1]);

    int i, j, k, x;
    FILE *fp;

	struct timespec start, stop; 
	double t;

	a = (unsigned char*) malloc (sizeof(unsigned char)*h*w);

	// the matrix is stored in a linear array in row major fashion
	if (!(fp=fopen(input_file, "rb"))) {
		printf("can not open file\n");
		return 1;
	}
	fread(a, sizeof(unsigned char), w*h, fp);
	fclose(fp);

    pthread_t threads[p];
	struct thread_data  thread_data_array[p];


	// measure the start time here
	if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}

	// 50 iterations
	for (i = 0; i < 50; i++)
	{
		// Iterating through every point
		for (j = 0; j < p; j++)
		{
            thread_data_array[j].thread_id = j;
            thread_data_array[j].size = h*w / p;
            thread_data_array[j].start = j * h*w / p;
			int rc = pthread_create(&threads[j], NULL, compute, (void *) &thread_data_array[j]);
			if (rc) { printf("ERROR; return code from pthread_create() is %d\n", rc); exit(-1);}
		}

        for (x = 0; x < p; x++)
        {
            pthread_join(threads[x], NULL);
        }
        

		// Iterating through each cluster to recalculate mean
		for (k = 0; k < K; k++)
		{
            int count = 0;
            int sum = 0;

            for (j = 0; j < p; j++)
            {
                count += thread_data_array[j].count[k];
                sum += thread_data_array[j].sums[k];
            }
            u[k] = sum / (double)count;
		}
	}

	// Updating the a[] array with cluster groups
	for (j = 0; j < h*w; j++)
	{
		// Finding the closest cluster to a point
		int min = fabs(a[j] - u[0]);
		unsigned char closest = 0;
		for (x = 1; x < K; x++)
		{
			int temp = fabs(a[j] - u[x]);
			if (temp < min) 
			{
				min =  temp;
				closest = x;
			}
		}

		a[j] = u[closest];
	}
	
	// measure the end time here
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
	t = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	
	// print out the execution time here
	printf("Time taken to run: %f sec\n", t);
	
	if (!(fp=fopen(output_file,"wb"))) {
		printf("can not opern file\n");
		return 1;
	}	
	fwrite(a, sizeof(unsigned char),w*h, fp);
    fclose(fp);

    free(a);
    
    pthread_exit(NULL);
    return 0;
}