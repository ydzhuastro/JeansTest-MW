CC = icc

make : 
	$(CC) -o directsum directsum.c -O3 -Wall -qopenmp -lgsl -lgslcblas -L/usr/local/lib -I/usr/local/include

clean : 
	rm directsum.o directsum