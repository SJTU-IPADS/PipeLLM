#!/bin/bash

rm -f /tmp/profile_iv.txt
LD_PRELOAD= nvcc ./iv.cu -lcuda -lpthread --cudart shared -o iv
./iv
