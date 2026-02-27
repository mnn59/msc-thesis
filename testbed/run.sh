#!/bin/bash
# SCENARIO=$1
sudo mn -c
for((i=1;i<=100;i++)) do sudo lsof -i:5000 | awk '{print $2}' | awk 'NR==2{print}' | xargs kill -9; done
# sudo python3 testbed.py $SCENARIO
sudo python3 testbed.py Abi
# sudo python testbed.py GEA
