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
    MPI_Bcast(numbers, ARY_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    int start, end;

    if(rank == 0){
       start = 0;
       end = 15;
    }	
    else if(rank == 1){
        start = 16;
        end = 31;
    }	
    else if(rank == 2){	
        start = 32;
        end = 47;
    }
    else if(rank == 3){
        start = 48;
        end = 63;
    }

    int partial_sum = 0;
    int total = 0;

    for (i = start; i <= end; i++)
    {
        partial_sum += numbers[i];
    }

    if (rank == 0) 
    {
        total += partial_sum;
        
        int temp;
        
        for (i = 1; i < 4; i++)
        {
            MPI_Recv(&temp, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total += temp;
        }

        printf("P2_2 Process 0: Total sum = %d\n", total);
    }
    else 
    {
        MPI_Send(&partial_sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}