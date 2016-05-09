obj = image_ppm.o sdr.o timer.o

CC = gcc
CFLAGS = -std=c89 -pedantic -Wall
LDFLAGS = -L/usr/X11R6/lib -lGL -lGLU -lglut -lm

.PHONY: all
all: mbrot julia

mbrot: mbrot.o $(obj)
	$(CC) -o $@ mbrot.o $(obj) $(LDFLAGS)

julia: julia.o $(obj)
	$(CC) -o $@ julia.o $(obj) $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(obj) mbrot.o julia.o mbrot julia
