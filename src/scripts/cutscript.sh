#!/bin/bash

ntrain=2339
ntest=2339

cat Tkernel.txt | cut -d' ' -f1-$ntrain | head -n -$ntest > kernel_train.txt
