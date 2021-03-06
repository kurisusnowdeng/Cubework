import copy
import math
from functools import partial
from typing import Callable

import cubework.module as cube_nn
import torch
from cubework.utils import get_current_device, get_dataloader, get_logger
from datasets import load_from_disk, set_progress_bar_enabled
from torch import dtype, nn
from transformers import GPT2Tokenizer
from transformers.optimization import get_linear_schedule_with_warmup


class GPT2Embedding(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        vocab_size: int,
        max_position_embeddings: int,
        num_tokentypes: int = 0,
        padding_idx: int = None,
        dropout: float = 0.0,
        dtype: dtype = None,
    ) -> None:
        super().__init__()
        self.word_embeddings = cube_nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx, dtype=dtype)
        self.position_embeddings = cube_nn.Embedding(max_position_embeddings, embedding_size, dtype=dtype)
        if num_tokentypes > 0:
            self.tokentype_embeddings = cube_nn.Embedding(num_tokentypes, embedding_size, dtype=dtype)
        else:
            self.tokentype_embeddings = None
        self.dropout = cube_nn.Dropout(dropout)

    @property
    def word_embedding_weight(self):
        return self.word_embeddings.weight

    def forward(self, input_ids, position_ids=None, tokentype_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=get_current_device()).unsqueeze(0)
        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        if self.tokentype_embeddings is not None and tokentype_ids is not None:
            x = x + self.tokentype_embeddings(tokentype_ids)
        x = self.dropout(x)

        return x


class GPT2SelfAttention(nn.Module):
    """Adapted from huggingface
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_positions: int,
        attention_dropout: float,
        dropout: float,
        bias: bool = True,
        dtype: dtype = None,
    ) -> None:
        super().__init__()
        self.attention_head_size = hidden_size // num_heads
        self.max_positions = max_positions
        self.query_key_value = cube_nn.Linear(hidden_size, 3 * hidden_size, dtype=dtype, bias=bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = cube_nn.Dropout(attention_dropout)
        self.dense = cube_nn.Linear(hidden_size, hidden_size, dtype=dtype, bias=True)
        self.dropout = cube_nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        qkv = self.query_key_value(x)
        all_head_size = qkv.shape[-1] // 3
        num_attention_heads = all_head_size // self.attention_head_size
        new_qkv_shape = qkv.shape[:-1] + (num_attention_heads, 3 * self.attention_head_size)
        qkv = qkv.view(new_qkv_shape)
        qkv = qkv.permute((0, 2, 1, 3))
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        x = torch.matmul(q, k.transpose(-1, -2))
        x = x / math.sqrt(self.attention_head_size)
        # causal mask
        causal_mask = (
            torch.tril(torch.ones((self.max_positions, self.max_positions), dtype=torch.uint8, device=x.device))
            .view(1, 1, self.max_positions, self.max_positions)
            .bool()
        )
        x = torch.where(causal_mask, x, torch.tensor(-1e4, dtype=x.dtype, device=x.device))
        if attention_mask is not None:
            x = x + attention_mask
        x = self.softmax(x)

        x = self.attention_dropout(x)

        x = torch.matmul(x, v)
        x = x.transpose(1, 2)
        new_context_layer_shape = x.size()[:-2] + (all_head_size,)
        x = x.reshape(new_context_layer_shape)

        x = self.dense(x)
        x = self.dropout(x)

        return x


class GPT2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: Callable,
        dropout: float,
        dtype: dtype = None,
        bias: bool = True,
    ):
        super().__init__()
        self.dense_1 = cube_nn.Linear(hidden_size, intermediate_size, dtype=dtype, bias=bias)
        self.activation = activation
        self.dense_2 = cube_nn.Linear(intermediate_size, hidden_size, dtype=dtype, bias=bias)
        self.dropout = cube_nn.Dropout(dropout)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class GPT2LMHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        word_embeeding_weight: nn.Parameter = None,
        bias: bool = False,
        dtype: dtype = None,
    ) -> None:
        super().__init__()
        self.dense = cube_nn.Classifier(
            hidden_size,
            vocab_size,
            weight=word_embeeding_weight,
            bias=bias,
            dtype=dtype,
        )

    def forward(self, x):
        x = self.dense(x)
        return x


class GPT2Block(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_positions: int,
        intermediate_size: int,
        activation: Callable,
        attention_dropout: float,
        dropout: float,
        layernorm_epsilon: float = 1e-5,
        dtype: dtype = None,
        bias: bool = True,
        apply_post_layernorm: bool = False,
        checkpoint: bool = False,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.apply_post_layernorm = apply_post_layernorm
        self.norm1 = cube_nn.LayerNorm(normalized_shape=hidden_size, eps=layernorm_epsilon, dtype=dtype)
        self.attn = GPT2SelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_positions=max_positions,
            attention_dropout=attention_dropout,
            dropout=dropout,
            bias=bias,
            dtype=dtype,
        )
        self.norm2 = cube_nn.LayerNorm(normalized_shape=hidden_size, eps=layernorm_epsilon, dtype=dtype)
        self.mlp = GPT2MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            dropout=dropout,
            dtype=dtype,
            bias=bias,
        )

    def _forward(self, x, attention_mask=None):
        if not self.apply_post_layernorm:
            residual = x
        x = self.norm1(x)
        if self.apply_post_layernorm:
            residual = x
        x = residual + self.attn(x, attention_mask)

        if not self.apply_post_layernorm:
            residual = x
        x = self.norm2(x)
        if self.apply_post_layernorm:
            residual = x
        x = residual + self.mlp(x)

        return x

    def forward(self, x, attention_mask=None):
        if self.checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, attention_mask)
        else:
            return self._forward(x, attention_mask)


class GPT2LMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = cube_nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50304,
        max_position_embeddings: int = 1024,
        hidden_size: int = 768,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        depth: int = 12,
        dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layernorm_epsilon: float = 1e-5,
        activation: Callable = nn.functional.gelu,
        padding_idx: int = None,
        dtype: dtype = None,
        bias: bool = True,
        apply_post_layernorm: bool = False,
        checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.embed = GPT2Embedding(
            embedding_size=hidden_size,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            padding_idx=padding_idx,
            dropout=embedding_dropout,
            dtype=dtype,
        )
        self.blocks = nn.ModuleList(
            [
                GPT2Block(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    max_positions=max_position_embeddings,
                    intermediate_size=intermediate_size,
                    activation=activation,
                    attention_dropout=attention_dropout,
                    dropout=dropout,
                    layernorm_epsilon=layernorm_epsilon,
                    dtype=dtype,
                    bias=bias,
                    apply_post_layernorm=apply_post_layernorm,
                    checkpoint=checkpoint,
                )
                for _ in range(depth)
            ]
        )

        self.norm = cube_nn.LayerNorm(normalized_shape=hidden_size, eps=layernorm_epsilon, dtype=dtype)

        self.head = GPT2LMHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            word_embeeding_weight=self.embed.word_embedding_weight,
            dtype=dtype,
        )

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # Adapted from huggingface
        if attention_mask is not None:
            batch_size = input_ids.shape[0]
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = cube_nn.partition_batch(attention_mask)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=x.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.head(self.norm(x))

        return x


def gpt2_small(checkpoint=False):
    model_kwargs = dict(
        hidden_size=768,
        intermediate_size=3072,
        depth=12,
        num_heads=16,
        checkpoint=checkpoint,
    )
    return GPT2(**model_kwargs)


def gpt2_medium(checkpoint=False):
    model_kwargs = dict(
        hidden_size=1024,
        intermediate_size=4096,
        depth=24,
        num_heads=16,
        checkpoint=checkpoint,
    )
    return GPT2(**model_kwargs)


def gpt2_large(checkpoint=True):
    model_kwargs = dict(
        hidden_size=1280,
        intermediate_size=5120,
        depth=36,
        num_heads=20,
        checkpoint=checkpoint,
    )
    return GPT2(**model_kwargs)


def gpt2_xl(checkpoint=True):
    model_kwargs = dict(
        hidden_size=1600,
        intermediate_size=6400,
        depth=48,
        num_heads=64,
        checkpoint=checkpoint,
    )
    return GPT2(**model_kwargs)


def gpt2_5b(checkpoint=True):
    model_kwargs = dict(
        hidden_size=2880,
        intermediate_size=11520,
        depth=50,
        num_heads=64,
        checkpoint=checkpoint,
    )
    return GPT2(**model_kwargs)


def gpt2_10b(checkpoint=True):
    model_kwargs = dict(
        hidden_size=4096,
        intermediate_size=16384,
        depth=50,
        num_heads=64,
        checkpoint=checkpoint,
    )
    return GPT2(**model_kwargs)


def gpt2_20b(checkpoint=True):
    model_kwargs = dict(
        hidden_size=5760,
        intermediate_size=23040,
        depth=50,
        num_heads=64,
        checkpoint=checkpoint,
    )
    return GPT2(**model_kwargs)


def gpt2_40b(checkpoint=True):
    model_kwargs = dict(
        hidden_size=8192,
        intermediate_size=32768,
        depth=50,
        num_heads=64,
        checkpoint=checkpoint,
    )
    return GPT2(**model_kwargs)


def build_model(args):
    model_func = globals()[args.model_name]
    return model_func(checkpoint=args.use_activation_checkpoint)


def _tokenize(examples, tokenizer):
    tokenizer.pad_token = tokenizer.unk_token
    examples = list(map(lambda x: x["text"], examples))
    result = tokenizer(examples, padding="max_length", truncation=True, max_length=1024, return_tensors="pt")
    result["labels"] = copy.deepcopy(result["input_ids"])
    return result


def build_data(args):
    set_progress_bar_enabled(False)
    dataset = load_from_disk(args.dataset_path)
    tokenizer = GPT2Tokenizer(
        vocab_file=args.tokenizer_path + "/vocab.json", merges_file=args.tokenizer_path + "/merges.txt"
    )

    train_data = get_dataloader(
        dataset=dataset["train"],
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,
        collate_fn=partial(_tokenize, tokenizer=tokenizer),
        num_workers=1,
        pin_memory=True,
    )
    test_data = get_dataloader(
        dataset=dataset["validation"],
        batch_size=args.batch_size,
        collate_fn=partial(_tokenize, tokenizer=tokenizer),
        num_workers=1,
        pin_memory=True,
    )

    return train_data, test_data


def build_criterion(args):
    return GPT2LMLoss(), cube_nn.Perplexity()


def build_optimizer(args, params):
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    return optimizer


def build_scheduler(args, n_steps, optimizer):
    max_steps = n_steps * args.num_epochs
    warmup_steps = n_steps * args.warmup_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
    )
    return lr_scheduler


def build_gpt2(args):
    logger = get_logger()

    model = build_model(args)
    logger.info("Model is built.")

    train_data, test_data = build_data(args)
    logger.info("Train and test data are built.")

    criterion, metric = build_criterion(args)
    logger.info("Loss and metric function are built.")

    optimizer = build_optimizer(args, model.parameters())
    n_steps = len(train_data)
    if args.steps_per_epoch is not None and args.steps_per_epoch < n_steps:
        n_steps = args.steps_per_epoch
    lr_scheduler = build_scheduler(args, n_steps, optimizer)
    logger.info("Optimizer and learning rate scheduler are built.")

    return model, train_data, test_data, criterion, metric, optimizer, lr_scheduler
