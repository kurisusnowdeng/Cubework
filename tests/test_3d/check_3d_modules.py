import time

import torch
from cubework.module import synchronize
from cubework.module.loss.loss_3d import CrossEntropyLoss3D, VocabParallelCrossEntropyLoss3D
from cubework.module.module_std import ClassifierSTD, PatchEmbeddingSTD
from cubework.module.parallel_3d import (Classifier3D, Embedding3D, LayerNorm3D, Linear3D, PatchEmbedding3D,
                                         VocabParallelClassifier3D, VocabParallelEmbedding3D)
from cubework.module.parallel_3d._utils import (get_input_parallel_mode, get_output_parallel_mode,
                                                get_weight_parallel_mode)
from cubework.utils import get_current_device, get_logger

DEPTH = 2
BATCH_SIZE = 8
SEQ_LENGTH = 8
HIDDEN_SIZE = 8
NUM_CLASSES = 8
NUM_BLOCKS = 2
IMG_SIZE = 16
VOCAB_SIZE = 16


def check_equal(A, B):
    eq = torch.allclose(A, B, rtol=1e-3, atol=1e-2)
    assert eq, f"\nA = {A}\nB = {B}"
    return eq


def check_linear():
    logger = get_logger()
    rank = torch.distributed.get_rank()

    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE
    OUTPUT_SIZE = 2 * HIDDEN_SIZE

    input_parallel_mode = get_input_parallel_mode()
    weight_parallel_mode = get_weight_parallel_mode()
    output_parallel_mode = get_output_parallel_mode()

    j = input_parallel_mode.local_rank
    i = weight_parallel_mode.local_rank
    k = output_parallel_mode.local_rank

    layer = Linear3D(INPUT_SIZE, OUTPUT_SIZE, dtype=dtype, bias=True)
    layer = layer.to(device)
    layer_master = torch.nn.Linear(INPUT_SIZE, OUTPUT_SIZE)
    layer_master = layer_master.to(device)

    weight_master = layer_master.weight.data.transpose(0, 1)
    torch.distributed.broadcast(weight_master, src=0)
    weight = torch.chunk(weight_master, DEPTH, dim=0)[k]
    weight = torch.chunk(weight, DEPTH, dim=-1)[j]
    weight = torch.chunk(weight, DEPTH, dim=-1)[i]
    layer.weight.data.copy_(weight)
    bias_master = layer_master.bias.data
    torch.distributed.broadcast(bias_master, src=0)
    bias = torch.chunk(bias_master, DEPTH)[j]
    layer.bias.data.copy_(bias)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, DEPTH, dim=0)[i]
    A = torch.chunk(A, DEPTH, dim=-1)[k]
    A = torch.chunk(A, DEPTH, dim=0)[j]
    A = A.clone()
    A.requires_grad = True

    fwd_start = time.time()
    out = layer(A)
    torch.cuda.synchronize()
    fwd_end = time.time()
    logger.info("linear forward: {0} --> {1} | {2:.3f} s".format(tuple(A.shape), tuple(out.shape), fwd_end - fwd_start))
    A_master = A_master.clone()
    A_master.requires_grad = True
    C_master = layer_master(A_master)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[j]
    C = torch.chunk(C, DEPTH, dim=0)[k]
    logger.info("Rank {} linear forward: {}".format(rank, check_equal(out, C)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_current_device())
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[j]
    grad = torch.chunk(grad, DEPTH, dim=0)[k]

    bwd_start = time.time()
    out.backward(grad)
    synchronize(layer.parameters())
    torch.cuda.synchronize()
    bwd_end = time.time()
    logger.info("linear backward: {:.3f} s".format(bwd_end - bwd_start))

    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[i]
    A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[k]
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[j]
    logger.info("Rank {} linear backward (input_grad): {}".format(rank, check_equal(A_grad, A.grad)))

    B_grad = layer_master.weight.grad.transpose(0, 1)
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[k]
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[j]
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[i]
    logger.info("Rank {} linear backward (weight_grad): {}".format(rank, check_equal(B_grad, layer.weight.grad)))

    bias_grad = layer_master.bias.grad
    bias_grad = torch.chunk(bias_grad, DEPTH)[j]
    logger.info("Rank {} linear backward (bias_grad): {}".format(rank, check_equal(bias_grad, layer.bias.grad)))

    return fwd_end - fwd_start, bwd_end - bwd_start


def check_layernorm():
    logger = get_logger()
    rank = torch.distributed.get_rank()

    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE

    input_parallel_mode = get_input_parallel_mode()
    weight_parallel_mode = get_weight_parallel_mode()
    output_parallel_mode = get_output_parallel_mode()

    j = input_parallel_mode.local_rank
    i = weight_parallel_mode.local_rank
    k = output_parallel_mode.local_rank

    norm = LayerNorm3D(INPUT_SIZE, eps=1e-6, dtype=dtype)
    norm = norm.to(device)
    norm_master = torch.nn.LayerNorm(INPUT_SIZE, eps=1e-6)
    norm_master = norm_master.to(device)

    weight_master = norm_master.weight.data
    torch.distributed.broadcast(weight_master, src=0)
    weight = torch.chunk(weight_master, DEPTH)[k]
    norm.weight.data.copy_(weight)
    bias_master = norm_master.bias.data
    torch.distributed.broadcast(bias_master, src=0)
    bias = torch.chunk(bias_master, DEPTH)[k]
    norm.bias.data.copy_(bias)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, DEPTH, dim=0)[i]
    A = torch.chunk(A, DEPTH, dim=-1)[k]
    A = torch.chunk(A, DEPTH, dim=0)[j]
    A = A.clone()
    A.requires_grad = True

    fwd_start = time.time()
    out = norm(A)
    torch.cuda.synchronize()
    fwd_end = time.time()
    logger.info("layer norm forward: pass | {0} --> {1} | {2:.3f} s".format(tuple(A.shape), tuple(out.shape),
                                                                            fwd_end - fwd_start))

    A_master = A_master.clone()
    A_master.requires_grad = True
    C_master = norm_master(A_master)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[k]
    C = torch.chunk(C, DEPTH, dim=0)[j]
    logger.info("Rank {} layernorm forward: {}".format(rank, check_equal(out, C)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[k]
    grad = torch.chunk(grad, DEPTH, dim=0)[j]

    bwd_start = time.time()
    out.backward(grad)
    synchronize(norm.parameters())
    torch.cuda.synchronize()
    bwd_end = time.time()
    logger.info("layer norm backward: pass | {:.3f} s".format(bwd_end - bwd_start))

    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[i]
    A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[k]
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[j]
    logger.info("Rank {} layernorm backward (input_grad): {}".format(rank, check_equal(A_grad, A.grad)))

    bias_grad = norm_master.weight.grad
    bias_grad = torch.chunk(bias_grad, DEPTH)[k]
    logger.info("Rank {} layernorm backward (weight_grad): {}".format(rank, check_equal(bias_grad, norm.weight.grad)))

    bias_grad = norm_master.bias.grad
    bias_grad = torch.chunk(bias_grad, DEPTH)[k]
    logger.info("Rank {} layernorm backward (bias_grad): {}".format(rank, check_equal(bias_grad, norm.bias.grad)))

    return fwd_end - fwd_start, bwd_end - bwd_start


def check_classifier_no_given_weight():
    logger = get_logger()
    rank = torch.distributed.get_rank()

    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE

    input_parallel_mode = get_input_parallel_mode()
    weight_parallel_mode = get_weight_parallel_mode()
    output_parallel_mode = get_output_parallel_mode()

    j = input_parallel_mode.local_rank
    i = weight_parallel_mode.local_rank
    k = output_parallel_mode.local_rank

    layer = Classifier3D(INPUT_SIZE, NUM_CLASSES, dtype=dtype, bias=True)
    layer = layer.to(device)

    layer_master = ClassifierSTD(INPUT_SIZE, NUM_CLASSES, bias=True, dtype=dtype)
    layer_master = layer_master.to(device)

    weight_master = layer_master.weight.data
    torch.distributed.broadcast(weight_master, src=0)
    weight = torch.chunk(weight_master, DEPTH, dim=-1)[k]
    layer.weight.data.copy_(weight)
    bias_master = layer_master.bias.data
    torch.distributed.broadcast(bias_master, src=0)
    layer.bias.data.copy_(bias_master)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, DEPTH, dim=0)[i]
    A = torch.chunk(A, DEPTH, dim=-1)[k]
    A = torch.chunk(A, DEPTH, dim=0)[j]
    A = A.clone()
    A.requires_grad = True

    fwd_start = time.time()
    out = layer(A)
    torch.cuda.synchronize()
    fwd_end = time.time()
    logger.info(
        "classifier (no given weight) forward: pass | {0} --> {1} | {2:.3f} s".format(
            tuple(A.shape), tuple(out.shape), fwd_end - fwd_start),)
    A_master = A_master.clone()
    A_master.requires_grad = True
    C_master = layer_master(A_master)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=0)[j]
    logger.info("Rank {} classifier (no given weight) forward: {}".format(rank, check_equal(out, C)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=get_current_device())
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=0)[j]
    grad = grad.clone()

    bwd_start = time.time()
    out.backward(grad)
    synchronize(layer.parameters())
    torch.cuda.synchronize()
    bwd_end = time.time()
    logger.info("classifier (no given weight) backward: pass | {:.3f} s".format(bwd_end - bwd_start))

    grad_master = grad_master.clone()
    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[i]
    A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[k]
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[j]
    logger.info("Rank {} classifier (no given weight) backward (input_grad): {}".format(
        rank, check_equal(A_grad, A.grad)))

    B_grad = layer_master.weight.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[k]
    if j == k:
        logger.info("Rank {} classifier (no given weight) backward (weight_grad): {}".format(
            rank, check_equal(B_grad, layer.weight.grad)))
    else:
        logger.info("Rank {} classifier (no given weight) backward (weight_grad): {}".format(
            rank, layer.weight.grad is None))

    bias_grad = layer_master.bias.grad
    logger.info("Rank {} classifier (no given weight) backward (bias_grad): {}".format(
        rank, check_equal(bias_grad, layer.bias.grad)))

    return fwd_end - fwd_start, bwd_end - bwd_start


def check_vocab_parallel_classifier_no_given_weight():
    logger = get_logger()
    rank = torch.distributed.get_rank()

    device = get_current_device()
    dtype = torch.float32
    INPUT_SIZE = HIDDEN_SIZE

    input_parallel_mode = get_input_parallel_mode()
    weight_parallel_mode = get_weight_parallel_mode()
    output_parallel_mode = get_output_parallel_mode()

    j = input_parallel_mode.local_rank
    i = weight_parallel_mode.local_rank
    k = output_parallel_mode.local_rank

    layer = VocabParallelClassifier3D(INPUT_SIZE, VOCAB_SIZE, bias=True)
    layer = layer.to(dtype).to(device)

    layer_master = ClassifierSTD(INPUT_SIZE, VOCAB_SIZE, bias=True)
    layer_master = layer_master.to(dtype).to(device)

    weight_master = layer_master.weight.data
    torch.distributed.broadcast(weight_master, src=0)
    weight = torch.chunk(weight_master, DEPTH, dim=0)[j]
    weight = torch.chunk(weight, DEPTH, dim=0)[i]
    weight = torch.chunk(weight, DEPTH, dim=-1)[k]
    layer.weight.data.copy_(weight)
    bias_master = layer_master.bias.data
    torch.distributed.broadcast(bias_master, src=0)
    bias = torch.chunk(bias_master, DEPTH)[j]
    layer.bias.data.copy_(bias)

    A_shape = (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = torch.chunk(A_master, DEPTH, dim=0)[i]
    A = torch.chunk(A, DEPTH, dim=-1)[k]
    A = torch.chunk(A, DEPTH, dim=0)[j]
    A = A.clone()
    A.requires_grad = True

    fwd_start = time.time()
    out = layer(A)
    torch.cuda.synchronize()
    fwd_end = time.time()
    logger.info(
        "vocab parallel classifier (no given weight) forward: pass | {0} --> {1} | {2:.3f} s".format(
            tuple(A.shape), tuple(out.shape), fwd_end - fwd_start),)
    A_master = A_master.clone()
    A_master.requires_grad = True
    C_master = layer_master(A_master)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[j]
    C = torch.chunk(C, DEPTH, dim=0)[k]
    logger.info("Rank {} vocab parallel classifier (no given weight) forward: {}".format(rank, check_equal(out, C)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[j]
    grad = torch.chunk(grad, DEPTH, dim=0)[k]
    grad = grad.clone()

    bwd_start = time.time()
    out.backward(grad)
    synchronize(layer.parameters())
    torch.cuda.synchronize()
    bwd_end = time.time()
    logger.info("vocab parallel classifier (no given weight) backward: pass | {:.3f} s".format(bwd_end - bwd_start))

    grad_master = grad_master.clone()
    C_master.backward(grad_master)
    A_grad = A_master.grad
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[i]
    A_grad = torch.chunk(A_grad, DEPTH, dim=-1)[k]
    A_grad = torch.chunk(A_grad, DEPTH, dim=0)[j]
    logger.info("Rank {} vocab parallel classifier (no given weight) backward (input_grad): {}".format(
        rank, check_equal(A_grad, A.grad)))

    B_grad = layer_master.weight.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[j]
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[i]
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[k]
    logger.info("Rank {} vocab parallel classifier (no given weight) backward (weight_grad): {}".format(
        rank, check_equal(B_grad, layer.weight.grad)))

    bias_grad = layer_master.bias.grad
    bias_grad = torch.chunk(bias_grad, DEPTH)[j]
    logger.info("Rank {} vocab parallel classifier (no given weight) backward (bias_grad): {}".format(
        rank, check_equal(bias_grad, layer.bias.grad)))

    return fwd_end - fwd_start, bwd_end - bwd_start


def check_vocab_parallel_classifier_given_embed_weight():
    logger = get_logger()
    rank = torch.distributed.get_rank()

    device = get_current_device()
    dtype = torch.float32

    input_parallel_mode = get_input_parallel_mode()
    weight_parallel_mode = get_weight_parallel_mode()
    output_parallel_mode = get_output_parallel_mode()

    j = input_parallel_mode.local_rank
    i = weight_parallel_mode.local_rank
    k = output_parallel_mode.local_rank

    embed = VocabParallelEmbedding3D(VOCAB_SIZE, HIDDEN_SIZE)
    embed = embed.to(dtype).to(device)

    embed_master = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    embed_master = embed_master.to(dtype).to(device)

    weight_master = embed_master.weight.data
    torch.distributed.broadcast(weight_master, src=0)
    weight = torch.chunk(weight_master, DEPTH, dim=0)[j]
    weight = torch.chunk(weight, DEPTH, dim=0)[i]
    weight = torch.chunk(weight, DEPTH, dim=-1)[k]
    embed.weight.data.copy_(weight)

    layer = VocabParallelClassifier3D(HIDDEN_SIZE, VOCAB_SIZE, weight=embed.weight, bias=False)
    layer = layer.to(dtype).to(device)

    layer_master = ClassifierSTD(HIDDEN_SIZE, VOCAB_SIZE, weight=embed_master.weight, bias=False)
    layer_master = layer_master.to(dtype).to(device)

    A_shape = (BATCH_SIZE, SEQ_LENGTH)
    A_master = torch.randint(VOCAB_SIZE, A_shape, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()

    fwd_start = time.time()
    out = layer(embed(A))
    torch.cuda.synchronize()
    fwd_end = time.time()
    logger.info(
        "vocab parallel classifier (given embed weight) forward: pass | {0} --> {1} | {2:.3f} s".format(
            tuple(A.shape), tuple(out.shape), fwd_end - fwd_start),)
    A_master = A_master.clone()
    C_master = layer_master(embed_master(A_master))
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[j]
    C = torch.chunk(C, DEPTH, dim=0)[k]
    logger.info("Rank {} vocab parallel classifier (given embed weight) forward: {}".format(rank, check_equal(out, C)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[j]
    grad = torch.chunk(grad, DEPTH, dim=0)[k]
    grad = grad.clone()

    bwd_start = time.time()
    out.backward(grad)
    synchronize(layer.parameters())
    synchronize(embed.parameters())
    torch.cuda.synchronize()
    bwd_end = time.time()
    logger.info("vocab parallel classifier (given embed weight) backward: pass | {:.3f} s".format(bwd_end - bwd_start))

    grad_master = grad_master.clone()
    C_master.backward(grad_master)

    B_grad = embed_master.weight.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[j]
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[i]
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[k]
    logger.info("Rank {} vocab parallel classifier (given embed weight) backward (weight_grad): {}".format(
        rank, check_equal(B_grad, embed.weight.grad)))

    return fwd_end - fwd_start, bwd_end - bwd_start


def check_patch_embed():
    logger = get_logger()
    rank = torch.distributed.get_rank()
    device = get_current_device()

    dtype = torch.float32

    input_parallel_mode = get_input_parallel_mode()
    weight_parallel_mode = get_weight_parallel_mode()
    output_parallel_mode = get_output_parallel_mode()

    j = input_parallel_mode.local_rank
    i = weight_parallel_mode.local_rank
    k = output_parallel_mode.local_rank

    layer = PatchEmbedding3D(IMG_SIZE, 4, 3, HIDDEN_SIZE, dtype=dtype)
    torch.nn.init.ones_(layer.cls_token)
    torch.nn.init.ones_(layer.pos_embed)
    layer = layer.to(device)

    layer_master = PatchEmbeddingSTD(IMG_SIZE, 4, 3, HIDDEN_SIZE, dtype=dtype)
    torch.nn.init.ones_(layer_master.cls_token)
    torch.nn.init.ones_(layer_master.pos_embed)
    layer_master = layer_master.to(device)

    proj_weight_master = layer_master.weight.data
    torch.distributed.broadcast(proj_weight_master, src=0)
    proj_weight = torch.chunk(proj_weight_master, DEPTH, dim=0)[k]
    layer.weight.data.copy_(proj_weight)
    proj_bias_master = layer_master.bias.data
    torch.distributed.broadcast(proj_bias_master, src=0)
    proj_bias = torch.chunk(proj_bias_master, DEPTH)[k]
    layer.bias.data.copy_(proj_bias)

    A_shape = (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)
    A_master = torch.randn(A_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()

    fwd_start = time.time()
    out = layer(A)
    torch.cuda.synchronize()
    fwd_end = time.time()
    logger.info(
        "patch embed forward: pass | {0} --> {1} | {2:.3f} s".format(tuple(A.shape), tuple(out.shape),
                                                                     fwd_end - fwd_start),)

    A_master = A_master.clone()
    C_master = layer_master(A_master)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[k]
    C = torch.chunk(C, DEPTH, dim=0)[j]
    logger.info("Rank {} patch embed forward: {}".format(rank, check_equal(out, C)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[k]
    grad = torch.chunk(grad, DEPTH, dim=0)[j]
    grad = grad.clone()

    bwd_start = time.time()
    out.backward(grad)
    synchronize(layer.parameters())
    torch.cuda.synchronize()
    bwd_end = time.time()
    logger.info("patch embed backward: pass | {:.3f} s".format(bwd_end - bwd_start))

    grad_master = grad_master.clone()
    C_master.backward(grad_master)

    cls_grad_master = layer_master.cls_token.grad
    cls_grad = torch.chunk(cls_grad_master, DEPTH, dim=-1)[k]
    logger.info("Rank {} patch embed backward (cls_grad): {}".format(rank, check_equal(cls_grad, layer.cls_token.grad)))

    pos_grad_master = layer_master.pos_embed.grad
    pos_grad = torch.chunk(pos_grad_master, DEPTH, dim=-1)[k]
    logger.info("Rank {} patch embed backward (pos_embed_grad): {}".format(rank,
                                                                           check_equal(pos_grad, layer.pos_embed.grad)))

    B_grad = layer_master.weight.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[k]
    logger.info("Rank {} patch embed backward (proj_weight_grad): {}".format(rank,
                                                                             check_equal(B_grad, layer.weight.grad)))

    bias_grad = layer_master.bias.grad
    bias_grad = torch.chunk(bias_grad, DEPTH)[k]
    logger.info("Rank {} patch embed backward (proj_bias_grad): {}".format(rank,
                                                                           check_equal(bias_grad, layer.bias.grad)))

    return fwd_end - fwd_start, bwd_end - bwd_start


def check_embed():
    logger = get_logger()
    rank = torch.distributed.get_rank()
    device = get_current_device()

    dtype = torch.float32

    input_parallel_mode = get_input_parallel_mode()
    weight_parallel_mode = get_weight_parallel_mode()
    output_parallel_mode = get_output_parallel_mode()

    j = input_parallel_mode.local_rank
    i = weight_parallel_mode.local_rank
    k = output_parallel_mode.local_rank

    layer = Embedding3D(VOCAB_SIZE, HIDDEN_SIZE)
    layer = layer.to(dtype).to(device)
    layer_master = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    layer_master = layer_master.to(dtype).to(device)

    weight_master = layer_master.weight.data
    torch.distributed.broadcast(weight_master, src=0)
    weight = torch.chunk(weight_master, DEPTH, dim=-1)[k]
    layer.weight.data.copy_(weight)

    A_shape = (BATCH_SIZE, SEQ_LENGTH)
    A_master = torch.randint(VOCAB_SIZE, A_shape, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()

    fwd_start = time.time()
    out = layer(A)
    torch.cuda.synchronize()
    fwd_end = time.time()
    logger.info("embed forward: pass | {0} --> {1} | {2:.3f} s".format(tuple(A.shape), tuple(out.shape),
                                                                       fwd_end - fwd_start))

    A_master = A_master.clone()
    C_master = layer_master(A_master)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[k]
    C = torch.chunk(C, DEPTH, dim=0)[j]
    logger.info("Rank {} embed forward: {}".format(rank, check_equal(out, C)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[k]
    grad = torch.chunk(grad, DEPTH, dim=0)[j]
    grad = grad.clone()
    bwd_start = time.time()
    out.backward(grad)
    synchronize(layer.parameters())
    torch.cuda.synchronize()
    bwd_end = time.time()
    logger.info("embed backward: pass | {:.3f} s".format(bwd_end - bwd_start))

    grad_master = grad_master.clone()
    C_master.backward(grad_master)

    B_grad = layer_master.weight.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[k]
    logger.info("Rank {} embed backward (weight_grad): {}".format(rank, check_equal(B_grad, layer.weight.grad)))

    return fwd_end - fwd_start, bwd_end - bwd_start


def check_vocab_parallel_embed():
    logger = get_logger()
    rank = torch.distributed.get_rank()
    device = get_current_device()

    dtype = torch.float32

    input_parallel_mode = get_input_parallel_mode()
    weight_parallel_mode = get_weight_parallel_mode()
    output_parallel_mode = get_output_parallel_mode()

    j = input_parallel_mode.local_rank
    i = weight_parallel_mode.local_rank
    k = output_parallel_mode.local_rank

    layer = VocabParallelEmbedding3D(VOCAB_SIZE, HIDDEN_SIZE)
    layer = layer.to(dtype).to(device)
    layer_master = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    layer_master = layer_master.to(dtype).to(device)

    weight_master = layer_master.weight.data
    torch.distributed.broadcast(weight_master, src=0)
    weight = torch.chunk(weight_master, DEPTH, dim=0)[j]
    weight = torch.chunk(weight, DEPTH, dim=0)[i]
    weight = torch.chunk(weight, DEPTH, dim=-1)[k]
    layer.weight.data.copy_(weight)

    A_shape = (BATCH_SIZE, SEQ_LENGTH)
    A_master = torch.randint(VOCAB_SIZE, A_shape, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()

    fwd_start = time.time()
    out = layer(A)
    torch.cuda.synchronize()
    fwd_end = time.time()
    logger.info("vocab parallel embed forward: pass | {0} --> {1} | {2:.3f} s".format(
        tuple(A.shape), tuple(out.shape), fwd_end - fwd_start))

    A_master = A_master.clone()
    C_master = layer_master(A_master)
    C = torch.chunk(C_master, DEPTH, dim=0)[i]
    C = torch.chunk(C, DEPTH, dim=-1)[k]
    C = torch.chunk(C, DEPTH, dim=0)[j]
    logger.info("Rank {} vocab parallel embed forward: {}".format(rank, check_equal(out, C)))

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = torch.chunk(grad_master, DEPTH, dim=0)[i]
    grad = torch.chunk(grad, DEPTH, dim=-1)[k]
    grad = torch.chunk(grad, DEPTH, dim=0)[j]
    grad = grad.clone()
    bwd_start = time.time()
    out.backward(grad)
    synchronize(layer.parameters())
    torch.cuda.synchronize()
    bwd_end = time.time()
    logger.info("vocab parallel embed backward: pass | {:.3f} s".format(bwd_end - bwd_start))

    grad_master = grad_master.clone()
    C_master.backward(grad_master)

    B_grad = layer_master.weight.grad
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[j]
    B_grad = torch.chunk(B_grad, DEPTH, dim=0)[i]
    B_grad = torch.chunk(B_grad, DEPTH, dim=-1)[k]
    logger.info("Rank {} vocab parallel embed backward (weight_grad): {}".format(rank,
                                                                                 check_equal(B_grad,
                                                                                             layer.weight.grad)))

    return fwd_end - fwd_start, bwd_end - bwd_start


def check_loss():
    logger = get_logger()
    rank = torch.distributed.get_rank()

    device = get_current_device()
    dtype = torch.float32

    input_parallel_mode = get_input_parallel_mode()
    weight_parallel_mode = get_weight_parallel_mode()

    j = input_parallel_mode.local_rank
    i = weight_parallel_mode.local_rank

    criterion = CrossEntropyLoss3D()
    criterion_master = torch.nn.CrossEntropyLoss()

    out_shape = (BATCH_SIZE, NUM_CLASSES)
    out_master = torch.randn(out_shape, dtype=dtype, device=device)
    target_master = torch.randint(NUM_CLASSES, (BATCH_SIZE,), dtype=torch.long, device=device)
    torch.distributed.broadcast(out_master, src=0)
    torch.distributed.broadcast(target_master, src=0)
    out = torch.chunk(out_master, DEPTH, dim=0)[i]
    out = torch.chunk(out, DEPTH, dim=0)[j]
    out = out.clone()
    out.requires_grad = True

    fwd_start = time.time()
    loss = criterion(out, target_master)
    fwd_end = time.time()
    logger.info("cross entropy loss forward: pass | {0} --> {1} | {2:.3f} s".format(tuple(out.shape), tuple(loss.shape),
                                                                                    fwd_end - fwd_start))

    out_master = out_master.clone()
    out_master.requires_grad = True
    loss_master = criterion_master(out_master, target_master)
    logger.info("Rank {} cross entropy loss forward: {}".format(rank, check_equal(loss, loss_master)))

    bwd_start = time.time()
    loss.backward()
    bwd_end = time.time()
    logger.info("cross entropy loss backward: pass | {:.3f} s".format(bwd_end - bwd_start))

    loss_master.backward()
    out_grad = out_master.grad
    out_grad = torch.chunk(out_grad, DEPTH, dim=0)[i]
    out_grad = torch.chunk(out_grad, DEPTH, dim=0)[j]
    logger.info("Rank {} cross entropy loss backward: {}".format(rank, check_equal(out_grad, out.grad)))

    return fwd_end - fwd_start, bwd_end - bwd_start


def check_vocab_parallel_loss():
    logger = get_logger()
    rank = torch.distributed.get_rank()

    device = get_current_device()
    dtype = torch.float32

    input_parallel_mode = get_input_parallel_mode()
    weight_parallel_mode = get_weight_parallel_mode()
    output_parallel_mode = get_output_parallel_mode()

    j = input_parallel_mode.local_rank
    i = weight_parallel_mode.local_rank
    k = output_parallel_mode.local_rank

    criterion = VocabParallelCrossEntropyLoss3D()
    criterion_master = torch.nn.CrossEntropyLoss()

    out_shape = (BATCH_SIZE, NUM_CLASSES)
    out_master = torch.randn(out_shape, dtype=dtype, device=device)
    target_master = torch.randint(NUM_CLASSES, (BATCH_SIZE,), dtype=torch.long, device=device)
    torch.distributed.broadcast(out_master, src=0)
    torch.distributed.broadcast(target_master, src=0)
    out = torch.chunk(out_master, DEPTH, dim=0)[i]
    out = torch.chunk(out, DEPTH, dim=-1)[k]
    out = torch.chunk(out, DEPTH, dim=0)[j]
    out = out.clone()
    out.requires_grad = True

    fwd_start = time.time()
    loss = criterion(out, target_master)
    fwd_end = time.time()
    logger.info("vocab parallel cross entropy loss forward: pass | {0} --> {1} | {2:.3f} s".format(
        tuple(out.shape), tuple(loss.shape), fwd_end - fwd_start))

    out_master = out_master.clone()
    out_master.requires_grad = True
    loss_master = criterion_master(out_master, target_master)
    logger.info("Rank {} vocab parallel cross entropy loss forward: {}".format(rank, check_equal(loss, loss_master)))

    bwd_start = time.time()
    loss.backward()
    bwd_end = time.time()
    logger.info("vocab parallel cross entropy loss backward: pass | {:.3f} s".format(bwd_end - bwd_start))

    loss_master.backward()
    out_grad = out_master.grad
    out_grad = torch.chunk(out_grad, DEPTH, dim=0)[i]
    out_grad = torch.chunk(out_grad, DEPTH, dim=-1)[k]
    out_grad = torch.chunk(out_grad, DEPTH, dim=0)[j]
    logger.info("Rank {} vocab parallel cross entropy loss backward: {}".format(rank, check_equal(out_grad, out.grad)))

    return fwd_end - fwd_start, bwd_end - bwd_start
