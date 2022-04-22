export WORLD_SIZE=$SLURM_NPROCS
export RANK=$SLURM_PROCID
export LOCAL_RANK=0
export MASTER_ADDR=mel${SLURM_NODELIST:4:4} 
export MASTER_PORT=23333

OMP_NUM_THREADS=12 \
NCCL_ALGO=Ring NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=ib0 \
NCCL_P2P_LEVEL=2 NCCL_NET_GDR_LEVEL=2 \
python train.py \
    --model gpt2_10b \
    --data $SCRATCH/dataset/huggingface/datasets/wikitext/wikitext-2/ \
    --token $SCRATCH/dataset/huggingface/tokenizers/gpt2/gpt2/ \
    --bs 16 --lr=0.001 --n_epoch 1 --n_step=3 \
    --amp --clip 1 --ckpt \
    --prof_mem --prof_comm \
    --tp 3d --tp_size 8
