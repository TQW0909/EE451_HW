#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ARY_SIZE 64

int main(int argc, char** argv)
{

    int size, rank;
    int numbers[ARY_SIZE];
    int i;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if (rank == 0)
    {
        // Reading in numbers from text
        FILE *file = fopen("number.txt", "r");
        for (i = 0; i < ARY_SIZE; i++)
        {
            fscanf(file, "%d", &numbers[i]);
        }
        fclose(file);
    }

    int local_numbers[ARY_SIZE / 4];

    MPI_Scatter(numbers, ARY_SIZE / 4, MPI_INT, &local_numbers, ARY_SIZE / 4, MPI_INT, 0, MPI_COMM_WORLD);

    int partial_sum = 0;
    int total = 0;

    for (i = 0; i < ARY_SIZE / 4; i++)
    {
        partial_sum += local_numbers[i];
    }

    int sums[4];
    MPI_Gather(&partial_sum, 1, MPI_INT, sums, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) 
    {        
        for (i = 0; i < 4; i++)
        {
            total += sums[i];
        }
        printf("P2_3 Process 0: Total sum = %d\n", total);
    }

    MPI_Finalize();

    return 0;
}