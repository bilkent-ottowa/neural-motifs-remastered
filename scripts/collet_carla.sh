#!/usr/bin/env bash

# This is a script that will evaluate all the models for SGDET
# export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=/home/yigityildirim/OpenAI/OpenAI\ non-IB/neural-motifs

python models/start_carla_collect.py --asynch False --numBurn 0 --save-dir ./data/carla_bev --save-name carla_bev.pkl \
    --numTicks 10 