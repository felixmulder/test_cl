CC=gcc
CFLAGS=-Wall -std=c99 -lm -lOpenCL
SOURCES=time_opencl.c
OUT=time

build: 	time_opencl.c
				$(CC) $(CFLAGS) $(SOURCES) -o $(OUT)

clean:
			rm -rf *.o $(OUT)
