# ViT benchmark

## Data parallel

```python
NUM_GPUS=8
MODEL=vit_base
DATASET=/PATH/TO/cifar-10/
BATCH_SIZE=128
LR=0.001
NUM_EPOCHS=200
WARMUP_STEPS=1000
VALIDATION_INTERVEL=10

torchrun --nproc_per_node $NUM_GPUS \
    train.py -m $MODEL --data $DATASET \
        --bs $BATCH_SIZE --lr $LR \
        --n_epoch $NUM_EPOCHS --n_warm $WARMUP_STEPS \
        --eval --n_eval $VALIDATION_INTERVEL \
        --clip 1.0 --amp --prof_comm
```

## Tensor parallel

```python
MODEL=vit_base
DATASET=/PATH/TO/cifar-10/
TENSOR_PARALLEL=3d
TP_SIZE=8
BATCH_SIZE=128
LR=0.001
NUM_EPOCHS=200
WARMUP_STEPS=1000
VALIDATION_INTERVEL=10

torchrun --nproc_per_node $NUM_GPUS \
    train.py -m $MODEL --data $DATASET\
        --tp $TENSOR_PARALLEL --tp_size $TP_SIZE \
        --bs $BATCH_SIZE --lr $LR \
        --n_epoch $NUM_EPOCHS --n_warm $WARMUP_STEPS \
        --eval --n_eval $VALIDATION_INTERVEL \
        --clip 1.0 --amp --prof_comm
```
