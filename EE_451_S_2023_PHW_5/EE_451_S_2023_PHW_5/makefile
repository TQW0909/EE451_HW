CC = nvcc

all: vector_add p1 p2

vector_add : vector_add.cu
	$(CC) vector_add.cu -o vector_add

p1 : p1.cu
	$(CC) p1.cu -o p1

p2 : p2.cu
	$(CC) p2.cu -o p2

clean:
	rm vector_add p1 p2