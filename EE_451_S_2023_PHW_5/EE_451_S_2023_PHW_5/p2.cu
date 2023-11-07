#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

#define n 1024
#define shared_size 32
#define block_size 32


__global__ void matrixMul(double *a, double *b, double *c){
	int x, y, my_id_x, my_id_y;
    x = threadIdx.x;
    y = threadIdx.y;
	my_id_x = blockIdx.x*blockDim.x + threadIdx.x;	
    my_id_y = blockIdx.y*blockDim.y + threadIdx.y;	
    
    double local_c = 0;

    __shared__ int a_shared[shared_size][shared_size];
    __shared__ int b_shared[shared_size][shared_size];

    for (int i = 0; i < n/block_size; i++)
    {
        a_shared[x][y] = a[my_id_x * n + (i * blockDim.y + y)];
        b_shared[x][y] = b[(i * blockDim.x + x) * n + my_id_y];
        __syncthreads();
        for (int j = 0; j < block_size; j++)
        {
            local_c += a_shared[x][j] * b_shared[j][y];
        }
        __syncthreads();
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
    
    dim3 dimGrid(32, 32);
    dim3 dimBlock(32, 32);
    
    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}

    matrixMul<<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c);				
    cudaMemcpy(c, gpu_c, sizeof(double)*n*n, cudaMemcpyDeviceToHost);
    
    if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	  
    time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
    printf("p2 time is %f ns\n", time*1e9);	 
    
    printf("c[%d][%d]=%f \n", 451, 451, c[451*n + 451]);

    free(a);
    free(b);
    free(c);
    cudaFree(gpu_a);  
    cudaFree(gpu_b);  
    cudaFree(gpu_c);  
    return 0;
}	
