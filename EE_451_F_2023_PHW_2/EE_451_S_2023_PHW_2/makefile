CC = gcc

all: p1a p1b p2

p1a : p1a.c
	$(CC) p1a.c -o p1a -lpthread

p1b : p1b.c
	$(CC) p1b.c -o p1b -lpthread

p2: p2.c
	$(CC) p2.c -o p2 -lpthread

clean:
	rm p1a p1b p2