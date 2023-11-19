#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

#define n 1024
#define blockSize 16 
#define gridSize 64

__global__ void matrixMul(double *a, double *b, double *c)
{
    int my_id_x, my_id_y;
    my_id_x = blockIdx.x * blockDim.x + threadIdx.x;
    my_id_y = blockIdx.y * blockDim.y + threadIdx.y;

    double local_c = 0;

    for (int i = 0; i < n; i++)
    {
        local_c += a[my_id_x * n + i] * b[i * n + my_id_y];
    }
    c[my_id_x * n + my_id_y] = local_c;
}

int main(int argc, char *argv[]){
		
	if (argc < 2)
	{
		printf("Need to enter number of nStreams!");
		return -1;
	}

	const int nStreams = atoi(argv[1]);
    const int streamSize = n*n / nStreams;
    const int streamBytes = streamSize * sizeof(double);

    int i, j;
    double *a = (double *)malloc(sizeof(double) * n * n);
    double *b = (double *)malloc(sizeof(double) * n * n);
    double *c = (double *)malloc(sizeof(double) * n * n);

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            a[i * n +j] = i;
            b[i * n +j] = j;
            c[i * n +j] = 0;
        }
    }

    cudaStream_t stream[nStreams];
    for (i = 0; i < nStreams; ++i)
    {
        cudaStreamCreate(&stream[i]);
    }

    double *gpu_a, *gpu_b, *gpu_c;
    cudaMalloc((void **)&gpu_a, sizeof(double) * n * n);
    cudaMalloc((void **)&gpu_b, sizeof(double) * n * n);
    cudaMalloc((void **)&gpu_c, sizeof(double) * n * n);

    cudaMemcpy(gpu_b, b, sizeof(double) * n * n, cudaMemcpyHostToDevice);

    dim3 dimGrid(gridSize / nStreams, gridSize);
    dim3 dimBlock(blockSize, blockSize);

    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    for (int i = 0; i < nStreams; ++i)
    {
        int offset = i * streamSize;
        cudaMemcpyAsync(&gpu_a[offset], &a[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]);
    }
    for (int i = 0; i < nStreams; ++i)
    {
        int offset = i * streamSize;
        matrixMul<<<dimGrid, dimBlock, 0, stream[i]>>>(gpu_a + offset, gpu_b, gpu_c + offset);
    }
    for (int i = 0; i < nStreams; ++i)
    {
        int offset = i * streamSize;
        cudaMemcpyAsync(&c[offset], &gpu_c[offset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);
    }

    for (i = 0; i < nStreams; ++i)
    {
        cudaStreamSynchronize(stream[i]);
    }
    
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);


    printf("p2 time is %f ms for nStreams = %d\n", time, nStreams);

    printf("c[%d][%d]=%f \n", 451, 451, c[451 * n + 451]);
    
    for (i = 0; i < nStreams; ++i)
    {
        cudaStreamDestroy(stream[i]);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    free(a);
    free(b);
    free(c);
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);
    return 0;
}
