RANK=$SLURM_PROCID LOCAL_RANK=0 WORLD_SIZE=$SLURM_NPROCS \
MASTER_ADDR=nid0${SLURM_NODELIST:5:4} MASTER_PORT=23333 \
NCCL_ALGO=Ring NCCL_CROSS_NIC=1 NCCL_P2P_LEVEL=2 NCCL_NET_GDR_LEVEL=2 \
python train.py \
    --model gpt2_xl \
    --data /data/scratch/huggingface/datasets/wikitext/wikitext-2/ \
    --token /data/scratch/huggingface/tokenizers/gpt2/gpt2/ \
    --bs 64 --lr=0.001 --n_epoch 1 --n_step=100 \
    --amp --clip 1 \
    --eval --prof_mem --prof_comm \
    --tp 1d --tp_size 4
