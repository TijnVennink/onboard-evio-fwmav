CC      = gcc
CFLAGS  = -O2 -fPIC -Wall -Wextra -I.
LDFLAGS = -lm

.PHONY: all clean

all: kalman_core_a.so

kalman_core_a.so: kalman_core_a_native.c kalman_core_a_native.h \
                  physicalConstants.h eventCameraConstants.h mat9.h
	$(CC) $(CFLAGS) -shared -o $@ kalman_core_a_native.c $(LDFLAGS)

clean:
	rm -f kalman_core_a.so
