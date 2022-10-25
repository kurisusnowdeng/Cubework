import torch
from cubework.distributed import ParallelManager as pm
from cubework.module.loss.loss_2d import CrossEntropyLoss2D, VocabParallelCrossEntropyLoss2D
from cubework.module.module_std import ClassifierSTD, PatchEmbeddingSTD
from cubework.module.parallel_2d import (
    Classifier2D,
    Embedding2D,
    LayerNorm2D,
    Linear2D,
    PatchEmbedding2D,
    VocabParallelClassifier2D,
    VocabParallelEmbedding2D,
)
from cubework.utils import get_current_device, get_logger

DEPTH = 2
BATCH_SIZE = 4
SEQ_LENGTH = 1
HIDDEN_SIZE = 8
NUM_CLASSES = 8
VOCAB_SIZE = 16
IMG_SIZE = 16


def check_equal(A, B):
    eq = torch.allclose(A, B, rtol=1e-3, atol=1e-2)
    assert eq
    return


def check_linear():
    logger = get_logger()
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE
    OUTPUT_SIZE = HIDDEN_SIZE

    j = pm.PARALLEL_2D_ROW.local_rank
    i = pm.PARALLEL_2D_COL.local_rank

    layer = Linear2D(INPUT_SIZE, OUTPUT_SIZE).to(device)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, DEPTH, dim=0)[i]
    A = torch.chunk(A, DEPTH, dim=-1)[j]
    A = A.clone()
    A.requires_grad = True

    W_shape = (INPUT_SIZE, OUTPUT_SIZE)
    W_master = torch.randn(W_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(W_master, src=0)
    W = torch.chunk(W_master, DEPTH, dim=0)[i]
    W = torch.chunk(W, DEPTH, dim=-1)[j]
    W = W.clone()
    W.requires_grad = True

    B_shape = OUTPUT_SIZE
    B_master = torch.randn(B_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(B_master, src=0)
    B = torch.chunk(B_master, DEPTH, dim=-1)[j]
    B = torch.chunk(B, DEPTH, dim=-1)[i]
    B = B.clone()
    B.requires_grad = True

    layer.weight.data.copy_(W)
    layer.bias.data.copy_(B)
    out = layer(A)

    A_master = A_master.clone()
    A_master.requires_grad = True
    W_master = W_master.clone()
    W_master.requires_grad = True
    B_master = B_master.clone()
    B_master.requires_grad = True
    C_master = torch.matmul(A_master, W_master) + B_master
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[j]

    check_equal(out, C)
    logger.info("linear forward: pass")

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_current_device())
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[j]
    grad = grad.clone()
    out.backward(grad)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[i]
    A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[j]
    check_equal(A_grad, A.grad)

    W_grad = W_master.grad
    W_grad = torch.chunk(W_grad, DEPTH, dim=0)[i]
    W_grad = torch.chunk(W_grad, DEPTH, dim=-1)[j]
    check_equal(W_grad, layer.weight.grad)

    B_grad = B_master.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[j]
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[i]
    # if i == 0:
    check_equal(B_grad, layer.bias.grad)

    logger.info("linear backward: pass")


def check_layernorm():
    logger = get_logger()
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE
    EPS = 1e-5

    j = pm.PARALLEL_2D_ROW.local_rank
    i = pm.PARALLEL_2D_COL.local_rank

    layernorm = LayerNorm2D(INPUT_SIZE).to(device)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, DEPTH, dim=0)[i]
    A = torch.chunk(A, DEPTH, dim=-1)[j]
    A = A.clone()
    A.requires_grad = True

    out = layernorm(A)

    A_master = A_master.clone()
    A_master.requires_grad = True
    E_master = torch.sum(A_master, dim=-1, keepdim=True)
    E_master /= INPUT_SIZE
    V_master = torch.sum(A_master * A_master, dim=-1, keepdim=True)
    V_master /= INPUT_SIZE
    V_master = V_master - E_master * E_master
    V_master = 1.0 / torch.sqrt(V_master + EPS)
    C_master = (A_master - E_master) * V_master
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[j]

    check_equal(out, C)
    logger.info("layer norm forward: pass")

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_current_device())
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[j]
    out.backward(grad)

    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[i]
    A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[j]
    check_equal(A_grad, A.grad)
    logger.info("layer norm backward: pass")


def check_embed():
    logger = get_logger()
    device = get_current_device()
    dtype = torch.float32
    j = pm.PARALLEL_2D_ROW.local_rank
    i = pm.PARALLEL_2D_COL.local_rank

    embed = Embedding2D(VOCAB_SIZE, HIDDEN_SIZE)
    embed = embed.to(dtype).to(device)
    embed_master = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    embed_master = embed_master.to(dtype).to(device)

    weight_master = embed_master.weight.data
    torch.distributed.broadcast(weight_master, src=0)
    weight = torch.chunk(weight_master, DEPTH, dim=-1)[j]
    weight = torch.chunk(weight, DEPTH, dim=-1)[i]
    embed.weight.data.copy_(weight)

    A_shape = (BATCH_SIZE, SEQ_LENGTH)
    A_master = torch.randint(VOCAB_SIZE, A_shape, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()
    out = embed(A)

    A_master = A_master.clone()
    C_master = embed_master(A_master)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[j]
    check_equal(out, C)
    logger.info("embed forward: pass")

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[j]
    grad = grad.clone()
    out.backward(grad)
    grad_master = grad_master.clone()
    C_master.backward(grad_master)

    B_grad = embed_master.weight.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[j]
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[i]
    check_equal(B_grad, embed.weight.grad)
    logger.info("embed backward: pass")


def check_patch_embed():
    logger = get_logger()
    device = get_current_device()
    dtype = torch.float32
    j = pm.PARALLEL_2D_ROW.local_rank
    i = pm.PARALLEL_2D_COL.local_rank

    layer = PatchEmbedding2D(IMG_SIZE, 4, 3, HIDDEN_SIZE)
    torch.nn.init.ones_(layer.cls_token)
    torch.nn.init.ones_(layer.pos_embed)
    layer = layer.to(device)

    layer_master = PatchEmbeddingSTD(IMG_SIZE, 4, 3, HIDDEN_SIZE)
    torch.nn.init.ones_(layer_master.cls_token)
    torch.nn.init.ones_(layer_master.pos_embed)
    layer_master = layer_master.to(device)

    proj_weight_master = layer_master.weight.data
    torch.distributed.broadcast(proj_weight_master, src=0)
    proj_weight = torch.chunk(proj_weight_master, DEPTH, dim=0)[j]
    proj_weight = torch.chunk(proj_weight, DEPTH, dim=0)[i]
    layer.weight.data.copy_(proj_weight)
    proj_bias_master = layer_master.bias.data
    torch.distributed.broadcast(proj_bias_master, src=0)
    proj_bias = torch.chunk(proj_bias_master, DEPTH, dim=0)[j]
    proj_bias = torch.chunk(proj_bias, DEPTH, dim=0)[i]
    layer.bias.data.copy_(proj_bias)

    A_shape = (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()
    out = layer(A)

    A_master = A_master.clone()
    C_master = layer_master(A_master)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[j]
    check_equal(out, C)
    logger.info("patch embed forward: pass")

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[j]
    grad = grad.clone()
    out.backward(grad)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)

    cls_grad_master = layer_master.cls_token.grad
    cls_grad = torch.chunk(cls_grad_master, DEPTH, dim=-1)[j]
    cls_grad = torch.chunk(cls_grad, DEPTH, dim=-1)[i]
    check_equal(cls_grad, layer.cls_token.grad)

    pos_grad_master = layer_master.pos_embed.grad
    pos_grad = torch.chunk(pos_grad_master, DEPTH, dim=-1)[j]
    pos_grad = torch.chunk(pos_grad, DEPTH, dim=-1)[i]
    check_equal(pos_grad, layer.pos_embed.grad)

    B_grad = layer_master.weight.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[j]
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[i]
    check_equal(B_grad, layer.weight.grad)

    bias_grad = layer_master.bias.grad
    bias_grad = torch.chunk(bias_grad, DEPTH)[j]
    bias_grad = torch.chunk(bias_grad, DEPTH)[i]
    check_equal(bias_grad, layer.bias.grad)
    logger.info("patch embed backward: pass")


def check_vocab_parallel_embed():
    logger = get_logger()
    device = get_current_device()
    dtype = torch.float32
    j = pm.PARALLEL_2D_ROW.local_rank
    i = pm.PARALLEL_2D_COL.local_rank

    embed = VocabParallelEmbedding2D(VOCAB_SIZE, HIDDEN_SIZE)
    embed = embed.to(dtype).to(device)
    embed_master = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    embed_master = embed_master.to(dtype).to(device)

    weight_master = embed_master.weight.data
    torch.distributed.broadcast(weight_master, src=0)
    weight = torch.chunk(weight_master, DEPTH, dim=-1)[j]
    weight = torch.chunk(weight, DEPTH, dim=0)[i]
    embed.weight.data.copy_(weight)

    A_shape = (BATCH_SIZE, SEQ_LENGTH)
    A_master = torch.randint(VOCAB_SIZE, A_shape, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()
    out = embed(A)

    A_master = A_master.clone()
    C_master = embed_master(A_master)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[j]
    check_equal(out, C)
    logger.info("vocab parallel embed forward: pass")

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[j]
    grad = grad.clone()
    out.backward(grad)
    grad_master = grad_master.clone()
    C_master.backward(grad_master)

    B_grad = embed_master.weight.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[j]
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[i]
    check_equal(B_grad, embed.weight.grad)
    logger.info("vocab parallel embed backward: pass")


def check_classifier_no_given_weight():
    logger = get_logger()
    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE
    OUTPUT_SIZE = NUM_CLASSES

    j = pm.PARALLEL_2D_ROW.local_rank
    i = pm.PARALLEL_2D_COL.local_rank

    layer = Classifier2D(INPUT_SIZE, OUTPUT_SIZE).to(device)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    A_master = torch.randint(5, A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, DEPTH, dim=0)[i]
    A = torch.chunk(A, DEPTH, dim=-1)[j]
    A = A.clone()
    A.requires_grad = True

    W_shape = (OUTPUT_SIZE, INPUT_SIZE)
    W_master = torch.randint(5, W_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(W_master, src=0)
    W = torch.chunk(W_master, DEPTH, dim=-1)[j]
    W = torch.chunk(W, DEPTH, dim=-1)[i]
    W = W.clone()
    layer.weight.data.copy_(W)
    # W.requires_grad = True

    B_shape = (OUTPUT_SIZE, )
    B_master = torch.randint(5, B_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(B_master, src=0)
    # B = torch.chunk(B_master, DEPTH, dim=0)[j]
    B = B_master.clone()
    layer.bias.data.copy_(B)

    out = layer(A)

    A_master = A_master.clone()
    A_master.requires_grad = True
    W_master = W_master.clone()
    W_master.requires_grad = True
    B_master = B_master.clone()
    B_master.requires_grad = True
    C_master = torch.matmul(A_master, W_master.transpose(0, 1)) + B_master
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    # C = torch.chunk(C, DEPTH, dim=-1)[j]

    check_equal(out, C)
    logger.info("classifier (no given weight) forward: pass")

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_current_device())
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    # grad = torch.chunk(grad, DEPTH, dim=-1)[j]
    grad = grad.clone()
    out.backward(grad)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[i]
    A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[j]
    check_equal(A_grad, A.grad)

    W_grad = W_master.grad
    W_grad = torch.chunk(W_grad, DEPTH, dim=-1)[j]
    W_grad = torch.chunk(W_grad, DEPTH, dim=-1)[i]
    check_equal(W_grad, layer.weight.grad)

    B_grad = B_master.grad
    # B_grad = torch.chunk(B_grad, DEPTH, dim=0)[j]
    # if i == 0:
    check_equal(B_grad, layer.bias.grad)

    logger.info("classifier (no given weight) backward: pass")


def check_vocab_parallel_classifier_no_given_weight():
    logger = get_logger()
    device = get_current_device()
    dtype = torch.float32

    j = pm.PARALLEL_2D_ROW.local_rank
    i = pm.PARALLEL_2D_COL.local_rank

    layer = VocabParallelClassifier2D(HIDDEN_SIZE, VOCAB_SIZE, bias=True)
    layer = layer.to(dtype).to(device)

    layer_master = ClassifierSTD(HIDDEN_SIZE, VOCAB_SIZE, bias=True)
    layer_master = layer_master.to(dtype).to(device)

    weight_master = layer_master.weight.data
    torch.distributed.broadcast(weight_master, src=0)
    weight = torch.chunk(weight_master, DEPTH, dim=0)[i]
    weight = torch.chunk(weight, DEPTH, dim=-1)[j]
    layer.weight.data.copy_(weight)
    bias_master = layer_master.bias.data
    torch.distributed.broadcast(bias_master, src=0)
    bias = torch.chunk(bias_master, DEPTH)[j]
    bias = torch.chunk(bias, DEPTH)[i]
    layer.bias.data.copy_(bias)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, DEPTH, dim=0)[i]
    A = torch.chunk(A, DEPTH, dim=-1)[j]
    A = A.clone()
    A.requires_grad = True
    out = layer(A)

    A_master = A_master.clone()
    A_master.requires_grad = True
    C_master = layer_master(A_master)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[j]
    check_equal(out, C)
    logger.info("vocab parallel classifier (no given weight) forward: pass")

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[j]
    grad = grad.clone()
    out.backward(grad)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[i]
    A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[j]
    check_equal(A_grad, A.grad)

    W_grad = layer_master.weight.grad
    W_grad = torch.chunk(W_grad, DEPTH, dim=0)[i]
    W_grad = torch.chunk(W_grad, DEPTH, dim=-1)[j]
    check_equal(W_grad, layer.weight.grad)

    B_grad = layer_master.bias.grad
    B_grad = torch.chunk(B_grad, DEPTH)[j]
    B_grad = torch.chunk(B_grad, DEPTH)[i]
    check_equal(B_grad, layer.bias.grad)
    logger.info("vocab parallel classifier (no given weight) backward: pass")


def check_classifier_given_embed_weight():
    logger = get_logger()
    device = get_current_device()
    dtype = torch.float32

    j = pm.PARALLEL_2D_ROW.local_rank
    i = pm.PARALLEL_2D_COL.local_rank

    embed = Embedding2D(VOCAB_SIZE, HIDDEN_SIZE)
    embed = embed.to(dtype).to(device)
    embed_master = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    embed_master = embed_master.to(dtype).to(device)

    weight_master = embed_master.weight.data
    torch.distributed.broadcast(weight_master, src=0)
    weight = torch.chunk(weight_master, DEPTH, dim=-1)[j]
    weight = torch.chunk(weight, DEPTH, dim=-1)[i]
    embed.weight.data.copy_(weight)

    layer = Classifier2D(HIDDEN_SIZE, VOCAB_SIZE, weight=embed.weight, bias=False)
    layer = layer.to(dtype).to(device)
    layer_master = ClassifierSTD(HIDDEN_SIZE, VOCAB_SIZE, weight=embed_master.weight, bias=False)
    layer_master = layer_master.to(dtype).to(device)

    A_shape = (BATCH_SIZE, SEQ_LENGTH)
    A_master = torch.randint(VOCAB_SIZE, A_shape, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()
    out = layer(embed(A))

    A_master = A_master.clone()
    C_master = layer_master(embed_master(A_master))
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    check_equal(out, C)
    logger.info("classifier (given embed weight) forward: pass")

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = grad.clone()
    out.backward(grad)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)

    W_grad = embed_master.weight.grad
    W_grad = torch.chunk(W_grad, DEPTH, dim=-1)[j]
    W_grad = torch.chunk(W_grad, DEPTH, dim=-1)[i]
    check_equal(W_grad, embed.weight.grad)
    logger.info("classifier (given embed weight) backward: pass")


def check_vocab_parallel_classifier_given_embed_weight():
    logger = get_logger()
    device = get_current_device()
    dtype = torch.float32

    j = pm.PARALLEL_2D_ROW.local_rank
    i = pm.PARALLEL_2D_COL.local_rank

    embed = VocabParallelEmbedding2D(VOCAB_SIZE, HIDDEN_SIZE)
    embed = embed.to(dtype).to(device)
    embed_master = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    embed_master = embed_master.to(dtype).to(device)

    weight_master = embed_master.weight.data
    torch.distributed.broadcast(weight_master, src=0)
    weight = torch.chunk(weight_master, DEPTH, dim=-1)[j]
    weight = torch.chunk(weight, DEPTH, dim=0)[i]
    embed.weight.data.copy_(weight)

    layer = VocabParallelClassifier2D(HIDDEN_SIZE, VOCAB_SIZE, weight=embed.weight, bias=False)
    layer = layer.to(dtype).to(device)
    layer_master = ClassifierSTD(HIDDEN_SIZE, VOCAB_SIZE, weight=embed_master.weight, bias=False)
    layer_master = layer_master.to(dtype).to(device)

    A_shape = (BATCH_SIZE, SEQ_LENGTH)
    A_master = torch.randint(VOCAB_SIZE, A_shape, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()
    out = layer(embed(A))

    A_master = A_master.clone()
    C_master = layer_master(embed_master(A_master))
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[j]
    check_equal(out, C)
    logger.info("vocab parallel classifier (given embed weight) forward: pass")

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[j]
    grad = grad.clone()
    out.backward(grad)

    grad_master = grad_master.clone()
    C_master.backward(grad_master)

    W_grad = embed_master.weight.grad
    W_grad = torch.chunk(W_grad, DEPTH, dim=-1)[j]
    W_grad = torch.chunk(W_grad, DEPTH, dim=0)[i]
    check_equal(W_grad, embed.weight.grad)
    logger.info("vocab parallel classifier (given embed weight) backward: pass")


def check_loss():
    logger = get_logger()
    device = get_current_device()
    dtype = torch.float32

    # j = pm.PARALLEL_2D_ROW.local_rank
    i = pm.PARALLEL_2D_COL.local_rank

    criterion = CrossEntropyLoss2D()
    criterion_master = torch.nn.CrossEntropyLoss()

    out_shape = (BATCH_SIZE, NUM_CLASSES)
    out_master = torch.randn(out_shape, dtype=dtype, device=device)
    target_master = torch.randint(NUM_CLASSES, (BATCH_SIZE, ), dtype=torch.long, device=device)
    torch.distributed.broadcast(out_master, src=0)
    torch.distributed.broadcast(target_master, src=0)
    out = torch.chunk(out_master, DEPTH, dim=0)[i]
    out = out.clone()
    out.requires_grad = True
    loss = criterion(out, target_master)

    out_master = out_master.clone()
    out_master.requires_grad = True
    loss_master = criterion_master(out_master, target_master)
    check_equal(loss, loss_master)
    logger.info("cross entropy loss forward: pass")

    loss.backward()
    loss_master.backward()

    out_grad = out_master.grad
    out_grad = torch.chunk(out_grad, DEPTH, dim=0)[i]
    check_equal(out_grad, out.grad)
    logger.info("cross entropy loss backward: pass")


def check_vocab_parallel_loss():
    logger = get_logger()
    device = get_current_device()
    dtype = torch.float32

    j = pm.PARALLEL_2D_ROW.local_rank
    i = pm.PARALLEL_2D_COL.local_rank

    criterion = VocabParallelCrossEntropyLoss2D()
    criterion_master = torch.nn.CrossEntropyLoss()

    out_shape = (BATCH_SIZE, NUM_CLASSES)
    out_master = torch.randn(out_shape, dtype=dtype, device=device)
    target_master = torch.randint(NUM_CLASSES, (BATCH_SIZE, ), dtype=torch.long, device=device)
    torch.distributed.broadcast(out_master, src=0)
    torch.distributed.broadcast(target_master, src=0)
    out = torch.chunk(out_master, DEPTH, dim=0)[i]
    out = torch.chunk(out, DEPTH, dim=-1)[j]
    out = out.clone()
    out.requires_grad = True
    loss = criterion(out, target_master)

    out_master = out_master.clone()
    out_master.requires_grad = True
    loss_master = criterion_master(out_master, target_master)
    check_equal(loss, loss_master)
    logger.info("vocab parallel cross entropy loss forward: pass")

    loss.backward()
    loss_master.backward()

    out_grad = out_master.grad
    out_grad = torch.chunk(out_grad, DEPTH, dim=0)[i]
    out_grad = torch.chunk(out_grad, DEPTH, dim=-1)[j]
    check_equal(out_grad, out.grad)
    logger.info("vocab parallel cross entropy loss backward: pass")
