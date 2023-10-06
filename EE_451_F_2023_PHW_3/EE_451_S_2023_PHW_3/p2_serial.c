#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define		size	    		2*1024*1024
#define		num_of_threads		2

void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void quickSort(int *array, int start, int end){
   // you quick sort function goes here
   // Middle pivot
   if (start < end) 
   {	
		int middle = start + ((end-start)/2);
		int pivot = array[middle];
		swap(&array[middle],&array[end]);
		int pivot_swap_index = start - 1;

		// Partitioning
		for (int i = start; i < end; i++)
		{
			if (array[i] < pivot)
			{
				// Swapping
				pivot_swap_index++;
				swap(&array[pivot_swap_index], &array[i]);
			}
		}
		// Swapping the pivot to the found position
		pivot_swap_index++;
		swap(&array[pivot_swap_index], &array[end]);
		
		quickSort(array, start, pivot_swap_index - 1);
		quickSort(array, pivot_swap_index + 1, end);
   }
}

int main(void){
	int i, j, tmp;
	struct timespec start, stop; 
	double exe_time;
	srand(time(NULL)); 
	int * m = (int *) malloc (sizeof(int)*size);
	for(i=0; i<size; i++){
		// m[i]=size-i;
		m[i] = rand();
	}
	
	if( clock_gettime(CLOCK_REALTIME, &start) == -1) { perror("clock gettime");}
	////////**********Your code goes here***************//
	
	// Serial
	quickSort(m, 0, size - 1);



	// // Parallel
	// int index = rand() % size;
	// int pivot = m[index];

	// swap(&m[index], &m[size - 1]);

	// int temp;
	// int pivot_swap_index = - 1;

	// // Partitioning
	// for (int i = 0; i < size - 1; i++)
	// {
	// 	if (m[i] < pivot)
	// 	{
	// 		// Swapping
	// 		pivot_swap_index++;
	// 		swap(&m[pivot_swap_index], &m[i]);
	// 	}
	// }
	// // Swapping the pivot to the found position
	// pivot_swap_index++;
	// swap(&m[pivot_swap_index], &m[size - 1]);
	
	// // Executing remainning using OpenMP
	// int tid;
	
	// omp_set_num_threads(num_of_threads);
	// #pragma omp parallel default(shared) private(tid)
	// {
	// 	tid = omp_get_thread_num();
	// 	# pragma omp sections
	// 	{
	// 		# pragma omp section
	// 		{
	// 			quickSort(m, 0, pivot_swap_index - 1);
	// 		}
	// 		# pragma omp section
	// 		{
	// 			quickSort(m, pivot_swap_index + 1, size - 1);
	// 		}
	// 	}		
	// }
			
	///////******************************////
	
	if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror("clock gettime");}		
	exe_time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
	
	for(i=0;i<16;i++) printf("%d ", m[i]);		
	printf("\nExecution time = %f sec\n",  exe_time);		

	free(m);
}	