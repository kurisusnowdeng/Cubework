import math
from functools import partial
from typing import Callable

import cubework.module as cube_nn
import numpy as np
import torch
import torchvision
from cubework.distributed import ParallelManager as pm
from cubework.module.utils import to_2tuple
from cubework.utils import get_current_device, get_dataloader, get_logger
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from transformers.optimization import get_cosine_schedule_with_warmup


class ViTEmbedding(nn.Module):

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embedding_size: int,
        dropout: float,
        flatten: bool = True,
    ):
        super().__init__()
        self.patch_embed = cube_nn.PatchEmbedding(img_size,
                                                  patch_size,
                                                  in_chans,
                                                  embedding_size,
                                                  flatten=flatten,
                                                  weight_initializer=cube_nn.init.lecun_normal_(),
                                                  bias_initializer=cube_nn.init.zeros_(),
                                                  position_embed_initializer=cube_nn.init.trunc_normal_(std=0.02))
        self.dropout = cube_nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.dropout(x)
        return x


class ViTSelfAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_dropout: float,
        dropout: float,
        bias: bool = True,
    ):
        super().__init__()
        self.attention_head_size = hidden_size // num_heads
        self.query_key_value = cube_nn.Linear(hidden_size,
                                              3 * hidden_size,
                                              bias=bias,
                                              weight_initializer=cube_nn.init.xavier_uniform_(),
                                              bias_initializer=cube_nn.init.normal_(std=1e-6))
        self.attention_dropout = cube_nn.Dropout(attention_dropout)
        self.dense = cube_nn.Linear(hidden_size,
                                    hidden_size,
                                    bias=True,
                                    weight_initializer=cube_nn.init.xavier_uniform_(),
                                    bias_initializer=cube_nn.init.normal_(std=1e-6))
        self.dropout = cube_nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        qkv = self.query_key_value(x)
        all_head_size = qkv.shape[-1] // 3
        num_attention_heads = all_head_size // self.attention_head_size
        new_qkv_shape = qkv.shape[:-1] + (num_attention_heads, 3 * self.attention_head_size)
        qkv = qkv.view(new_qkv_shape)
        qkv = qkv.permute((0, 2, 1, 3))
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        x = torch.matmul(q, k.transpose(-1, -2))
        x = x / math.sqrt(self.attention_head_size)
        x = self.softmax(x)
        x = self.attention_dropout(x)

        x = torch.matmul(x, v)
        x = x.transpose(1, 2)
        new_context_layer_shape = x.size()[:-2] + (all_head_size, )
        x = x.reshape(new_context_layer_shape)

        x = self.dense(x)
        x = self.dropout(x)

        return x


class ViTMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: Callable,
        dropout: float,
        bias: bool = True,
    ):
        super().__init__()
        self.dense_1 = cube_nn.Linear(hidden_size,
                                      intermediate_size,
                                      bias=bias,
                                      weight_initializer=cube_nn.init.xavier_uniform_(),
                                      bias_initializer=cube_nn.init.normal_(std=1e-6))
        self.activation = activation
        self.dropout_1 = cube_nn.Dropout(dropout)
        self.dense_2 = cube_nn.Linear(intermediate_size,
                                      hidden_size,
                                      bias=bias,
                                      weight_initializer=cube_nn.init.xavier_uniform_(),
                                      bias_initializer=cube_nn.init.normal_(std=1e-6))
        self.dropout_2 = cube_nn.Dropout(dropout)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = self.dropout_2(x)
        return x


class ViTHead(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        representation_size: int = None,
        bias: bool = True,
    ):
        super().__init__()
        if representation_size:
            self.representation = cube_nn.Linear(hidden_size,
                                                 representation_size,
                                                 bias=bias,
                                                 weight_initializer=cube_nn.init.zeros_(),
                                                 bias_initializer=cube_nn.init.zeros_())
        else:
            self.representation = None
            representation_size = hidden_size

        self.dense = cube_nn.Classifier(representation_size,
                                        num_classes,
                                        bias=bias,
                                        weight_initializer=cube_nn.init.zeros_(),
                                        bias_initializer=cube_nn.init.zeros_())

    def forward(self, x):
        x = x[:, 0]
        if self.representation is not None:
            x = self.representation(x)
        x = self.dense(x)
        return x


class ViTBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        activation: Callable,
        attention_dropout: float,
        dropout: float,
        drop_path: float = 0.0,
        layernorm_epsilon: float = 1e-6,
        bias: bool = True,
        checkpoint: bool = False,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.norm1 = cube_nn.LayerNorm(normalized_shape=hidden_size, eps=layernorm_epsilon)
        self.attn = ViTSelfAttention(hidden_size=hidden_size,
                                     num_heads=num_heads,
                                     attention_dropout=attention_dropout,
                                     dropout=dropout,
                                     bias=bias)
        self.drop_path = cube_nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = cube_nn.LayerNorm(normalized_shape=hidden_size, eps=layernorm_epsilon)
        self.mlp = ViTMLP(hidden_size=hidden_size,
                          intermediate_size=intermediate_size,
                          activation=activation,
                          dropout=dropout,
                          bias=bias)

    def _forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x):
        if self.checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)


class VisionTransformer(nn.Module):

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        depth: int = 12,
        hidden_size: int = 768,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        representation_size: int = None,
        attention_dropout: float = 0.0,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        layernorm_epsilon: float = 1e-6,
        activation: Callable = nn.functional.gelu,
        bias: bool = True,
        checkpoint: bool = False,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = grid_size[0] * grid_size[1]

        self.embed = ViTEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embedding_size=hidden_size,
            dropout=dropout,
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            ViTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                attention_dropout=attention_dropout,
                dropout=dropout,
                drop_path=dpr[i],
                activation=activation,
                bias=bias,
                checkpoint=checkpoint,
            ) for i in range(depth)
        ])

        self.norm = cube_nn.LayerNorm(normalized_shape=hidden_size, eps=layernorm_epsilon)

        self.head = ViTHead(hidden_size=hidden_size,
                            num_classes=num_classes,
                            representation_size=representation_size,
                            bias=bias)

    def forward(self, pixel_values):
        x = self.embed(pixel_values)

        for block in self.blocks:
            x = block(x)

        x = self.head(self.norm(x))

        return x


def vit_small(checkpoint=False):
    model_kwargs = dict(
        img_size=224,
        patch_size=16,
        hidden_size=384,
        num_heads=16,
        intermediate_size=1536,
        depth=12,
        checkpoint=checkpoint,
    )
    return VisionTransformer(**model_kwargs)


def vit_base(checkpoint=False):
    model_kwargs = dict(
        img_size=224,
        patch_size=16,
        hidden_size=768,
        num_heads=16,
        intermediate_size=3072,
        depth=12,
        checkpoint=checkpoint,
    )
    return VisionTransformer(**model_kwargs)


def vit_g(checkpoint=True):
    model_kwargs = dict(
        img_size=224,
        patch_size=14,
        hidden_size=1664,
        num_heads=64,
        intermediate_size=8192,
        depth=48,
        checkpoint=checkpoint,
    )
    return VisionTransformer(**model_kwargs)


def vit_3b(checkpoint=True):
    model_kwargs = dict(
        img_size=224,
        patch_size=14,
        hidden_size=1920,
        num_heads=64,
        intermediate_size=7680,
        depth=64,
        checkpoint=checkpoint,
    )
    return VisionTransformer(**model_kwargs)


def vit_6b(checkpoint=True):
    model_kwargs = dict(
        img_size=224,
        patch_size=14,
        hidden_size=2560,
        num_heads=64,
        intermediate_size=10240,
        depth=72,
        checkpoint=checkpoint,
    )
    return VisionTransformer(**model_kwargs)


def vit_12b(checkpoint=True):
    model_kwargs = dict(
        img_size=224,
        patch_size=14,
        hidden_size=3200,
        num_heads=64,
        intermediate_size=12800,
        depth=96,
        checkpoint=checkpoint,
    )
    return VisionTransformer(**model_kwargs)


def _mixup_data(features, transform, alpha=0.0, train=True):
    x, y = tuple(zip(*features))
    x = torch.stack([transform(i) for i in x])
    y = torch.as_tensor(y)

    if train:
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        lam = torch.tensor(lam).to(mixed_x.dtype)

        return {
            "pixel_values": mixed_x,
            "labels": {
                "y_a": y_a,
                "y_b": y_b,
                "lam": lam,
            },
        }

    else:
        return {
            "pixel_values": x,
            "labels": {
                "y_a": y,
                "y_b": y,
                "lam": torch.ones(()).to(x.dtype),
            },
        }


class MixupLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = cube_nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        y_a, y_b, lam = targets["y_a"], targets["y_b"], targets["lam"]
        return lam * self.loss_fn(inputs, y_a) + (1 - lam) * self.loss_fn(inputs, y_b)


class MixupAccuracy(cube_nn.Accuracy):

    def forward(self, logits, targets, loss):
        targets = targets["y_a"]
        return super().forward(logits, targets, loss)


def build_model(args):
    logger = get_logger()
    model_func = globals()[args.model_name]
    model = model_func(checkpoint=args.use_activation_checkpoint)
    model = model.to(get_current_device())

    if pm.DATA.world_size > 1:
        model = DDP(model, process_group=pm.DATA.group)
        setattr(model, "num_patches", model.module.num_patches)

    logger.info("Model is built.")
    return model


def build_data(args):
    logger = get_logger()
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.micro_batch_size is None:
        args.micro_batch_size = args.global_batch_size // pm.DATA.world_size
    logger.info(f"Using global batch size = {args.global_batch_size}, micro batch size = {args.micro_batch_size}.")

    train_dataset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=False)
    train_data = get_dataloader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=args.micro_batch_size,
        drop_last=True,
        collate_fn=partial(_mixup_data, transform=transform_train, alpha=0.8, train=True),
        num_workers=4,
        pin_memory=True,
    )
    test_data = get_dataloader(
        dataset=test_dataset,
        batch_size=args.micro_batch_size,
        collate_fn=partial(_mixup_data, transform=transform_test, alpha=0.8, train=False),
        num_workers=4,
        pin_memory=True,
    )
    logger.info("Train and test data are built.")
    return train_data, test_data


def build_criterion():
    logger = get_logger()
    criterion, metric = MixupLoss(), MixupAccuracy()
    logger.info("Loss and metric function are built.")
    return criterion, metric


def build_optimizer(args, params):
    logger = get_logger()
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    logger.info("Optimizer is built.")
    return optimizer


def build_scheduler(args, n_steps, optimizer):
    logger = get_logger()
    max_steps = n_steps * args.num_epochs
    warmup_steps = min(args.warmup_steps, max_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=warmup_steps,
                                                   num_training_steps=max_steps)
    logger.info("Learning rate scheduler is built.")
    return lr_scheduler


def build_vit(args):
    model = build_model(args)

    train_data, test_data = build_data(args)

    criterion, metric = build_criterion()

    optimizer = build_optimizer(args, model.parameters())
    n_steps = len(train_data)
    if args.steps_per_epoch is not None and args.steps_per_epoch < n_steps:
        n_steps = args.steps_per_epoch
    lr_scheduler = build_scheduler(args, n_steps, optimizer)

    return model, train_data, test_data, criterion, metric, optimizer, lr_scheduler
