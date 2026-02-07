#!/bin/bash

P=$1
K=$2
W=$3
BETA2=$4
BETA4=$5
NEVENTS=100000

trento Pb Pb \
  --number-events $NEVENTS \
  --p $P \
  --fluctuation $K \
  --nucleon-width $W \
  --beta2 $BETA2 \
  --beta4 $BETA4 \
  --output trento.h5
