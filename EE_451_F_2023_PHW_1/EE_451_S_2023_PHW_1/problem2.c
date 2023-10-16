#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define h  800 
#define w  800
#define K 4

#define input_file  "input.raw"
#define output_file "output.raw"

int main(int argc, char** argv){
    int i, j, k, x;
    FILE *fp;

	struct timespec start, stop; 
	double t;

	unsigned char u[K] = {0, 85, 170, 255}; // Initial cluster mean values

	unsigned char *a = (unsigned char*) malloc (sizeof(unsigned char)*h*w);

	// Struct for each of the 4 clusters
	typedef struct {
		int num_points;
		unsigned char mean;
		unsigned char* elements;
	} Cluster;
    
	// the matrix is stored in a linear array in row major fashion
	if (!(fp=fopen(input_file, "rb"))) {
		printf("can not open file\n");
		return 1;
	}
	fread(a, sizeof(unsigned char), w*h, fp);
	fclose(fp);
    
	// measure the start time here
	if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}

	//  Your code goes here
	
	// Array of clusters
	Cluster* clusters = (Cluster*) malloc (sizeof(Cluster) * K);

	// Initializing each cluster
	for (i = 0; i < K; i++)
	{
		clusters[i].num_points = 0;
		clusters[i].mean = u[i];
		clusters[i].elements = (unsigned char*) malloc (sizeof(unsigned char)*h*w);
	}
	
	// 30 iterations
	for (i = 0; i < 30; i++)
	{
		// Iterating through every point
		for (j = 0; j < h*w; j++)
		{
			// Finding the clostest cluster to a point
			int min = abs(a[j] - clusters[0].mean);
			unsigned char closest = 0;
			for (x = 1; x < K; x++)
			{
				unsigned char temp = abs(a[j] - clusters[x].mean);
				if (temp < min) 
				{
					min =  temp;
					closest = x;
				}
			}

			clusters[closest].elements[clusters[closest].num_points] = a[j]; // Assigning point to closest cluster
			clusters[closest].num_points++;
		}

		// Iterating through each cluster to recalculate mean
		for (k = 0; k < K; k++)
		{
			if (clusters[k].num_points != 0)
			{
				int mean = 0;
				for (x = 0; x < clusters[k].num_points; x++) 
				{
					mean += clusters[k].elements[x];
				}
				mean /= clusters[k].num_points;
				clusters[k].mean = mean;
				clusters[k].num_points = 0; // Resetting points belong to each cluster
			}
		}
	}

	// Updating the a[] array with cluster groups
	for (j = 0; j < h*w; j++)
	{
		// Finding the closest cluster to a point
		int min = abs(a[j] - clusters[0].mean);
		unsigned char closest = 0;
		for (x = 1; x < K; x++)
		{
			int temp = abs(a[j] - clusters[x].mean);
			if (temp < min) 
			{
				min =  temp;
				closest = x;
			}
		}

		a[j] = clusters[closest].mean;
	}
		
	//
	
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

	// Freeing memory
	free(a);
	for (i = 0; i < K; i++) 
	{
		free(clusters[i].elements);
	}
	free(clusters);
    
    return 0;
}