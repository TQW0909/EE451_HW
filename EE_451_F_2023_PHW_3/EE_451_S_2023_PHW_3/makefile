CC = gcc-13 -fopenmp 

all: p1a p1b p1_serial p2 p2_serial p3

p1_serial : p1_serial.c
	$(CC) p1_serial.c -o p1_serial

p1a : p1a.c
	$(CC) p1a.c -o p1a

p1b : p1b.c
	$(CC) p1b.c -o p1b

p2: p2.c
	$(CC) p2.c -o p2

p2_serial: p2_serial.c
	$(CC) p2_serial.c -o p2_serial

p3: p3.c
	$(CC) p3.c -o p3


clean:
	rm p1a p1b p1_serial p2 p2_serial p3