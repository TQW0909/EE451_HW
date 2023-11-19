CC = nvcc

all: async p1 p2

async : async.cu
	$(CC) async.cu -o async

p1 : p1.cu
	$(CC) p1.cu -o p1

p2 : p2.cu
	$(CC) p2.cu -o p2

clean:
	rm async p1 p2