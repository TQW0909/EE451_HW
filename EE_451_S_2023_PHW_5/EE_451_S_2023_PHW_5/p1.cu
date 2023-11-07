#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

#define n 1024


__global__ void matrixMul(double *a, double *b, double *c){
	int my_id_x, my_id_y;
	my_id_x = blockIdx.x*blockDim.x + threadIdx.x;	
    my_id_y = blockIdx.y*blockDim.y + threadIdx.y;	
    
    double local_c = 0;

    for (int i = 0; i < n; i++)
    {
        local_c += a[my_id_x * n + i] * b[i * n + my_id_y];
    }
	c[my_id_x * n + my_id_y] = local_c;  
}

int main(){		
    int i;
    double *a = (double*)malloc(sizeof(double)*n*n);   
    double *b = (double*)malloc(sizeof(double)*n*n);
    double *c = (double*)malloc(sizeof(double)*n*n);       
    
    for (i = 0; i < n * n; i++)
    {
        a[i] = 1;
        b[i] = 2;
        c[i] = 0;
    }
         	

    double *gpu_a, *gpu_b, *gpu_c;
    cudaMalloc((void**)&gpu_a, sizeof(double)*n*n); 
    cudaMalloc((void**)&gpu_b, sizeof(double)*n*n);
    cudaMalloc((void**)&gpu_c, sizeof(double)*n*n);
    
    struct timespec start, stop; 
    double time;
    
    cudaMemcpy(gpu_a, a, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    
    dim3 dimGrid(64, 64);
    dim3 dimBlock(16, 16);
    
    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}

    matrixMul<<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c);				
    cudaMemcpy(c, gpu_c, sizeof(double)*n*n, cudaMemcpyDeviceToHost);
    
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	  
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    printf("p1 time is %f ns\n", time*1e9);	 
    
    printf("c[%d][%d]=%f \n", 451, 451, c[451*n + 451]);

    free(a);
    free(b);
    free(c);
    cudaFree(gpu_a);  
    cudaFree(gpu_b);  
    cudaFree(gpu_c);  
    return 0;
}	
