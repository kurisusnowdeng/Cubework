# OPT benchmark

## Data parallel

```python
NUM_GPUS=8
MODEL=opt_125m
SEQ_LENGTH=512
DATASET=/PATH/TO/openwebtext/
TOKENIZER=./
BATCH_SIZE=128
LR=0.001
NUM_EPOCHS=20
WARMUP_STEPS=1000
VALIDATION_INTERVEL=4

torchrun --nproc_per_node $NUM_GPUS \
    train.py -m $MODEL -s $SEQ_LENGTH \
        --data $DATASET --tok $TOKENIZER \
        --bs $BATCH_SIZE --lr $LR \
        --n_epoch $NUM_EPOCHS --n_warm $WARMUP_STEPS \
        --eval --n_eval $VALIDATION_INTERVEL \
        --clip 1.0 --amp --prof_comm
```

## Tensor parallel

```python
NUM_GPUS=8
MODEL=opt_125m
SEQ_LENGTH=512
TENSOR_PARALLEL=3d
TP_SIZE=8
DATASET=/PATH/TO/openwebtext/
TOKENIZER=./
BATCH_SIZE=128
LR=0.001
NUM_EPOCHS=20
WARMUP_STEPS=1000
VALIDATION_INTERVEL=4

torchrun --nproc_per_node $NUM_GPUS \
    train.py -m $MODEL -s $SEQ_LENGTH \
        --tp $TENSOR_PARALLEL --tp_size $TP_SIZE \
        --data $DATASET --tok $TOKENIZER \
        --bs $BATCH_SIZE --lr $LR \
        --n_epoch $NUM_EPOCHS --n_warm $WARMUP_STEPS \
        --eval --n_eval $VALIDATION_INTERVEL \
        --clip 1.0 --amp --prof_comm
```

## Tensor parallel + FSDP

```python
NUM_GPUS=8
MODEL=opt_125m
SEQ_LENGTH=512
TENSOR_PARALLEL=1d
TP_SIZE=2
DATASET=/PATH/TO/openwebtext/
TOKENIZER=./
BATCH_SIZE=128
LR=0.001
NUM_EPOCHS=20
WARMUP_STEPS=1000
VALIDATION_INTERVEL=4

torchrun --nproc_per_node $NUM_GPUS \
    train.py -m $MODEL -s $SEQ_LENGTH \
        --tp $TENSOR_PARALLEL --tp_size $TP_SIZE \
        --data $DATASET --tok $TOKENIZER \
        --bs $BATCH_SIZE --lr $LR \
        --n_epoch $NUM_EPOCHS --n_warm $WARMUP_STEPS \
        --eval --n_eval $VALIDATION_INTERVEL \
        --clip 1.0 --amp --fsdp --prof_comm
```
