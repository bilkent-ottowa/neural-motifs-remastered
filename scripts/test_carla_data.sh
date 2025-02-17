#!/usr/bin/env bash
export PYTHONPATH=/home/yigityildirim/OpenAI/OpenAI\ non-IB/neural-motifs

echo "EVALING ON CARLA DATASET"
python models/test_carla_dataloader.py -m predcls -model motifnet -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/vgrel-motifnet-sgcls.tar -nepoch 50 -cache motifnet_predcls_carla