import random
from typing import Callable, List, Optional, Tuple

import cubework.module as cube_nn
from functools import partial
import torch
from cubework.utils import get_dataloader, get_logger
from datasets import load_from_disk
from torch import nn
from transformers import GPT2Tokenizer
from transformers.optimization import get_linear_schedule_with_warmup


class OPTLearnedPositionalEmbedding(cube_nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # self.num_heads = num_heads
        # self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                             f" and `num_heads`: {num_heads}).")
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = cube_nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        # self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = cube_nn.Dropout(dropout)
        self.out_proj = cube_nn.Linear(embed_dim, embed_dim, bias=bias)

    # def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
    #     return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
    ):
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, _ = hidden_states.size()

        qkv_states = self.qkv_proj(hidden_states)
        num_heads = qkv_states.shape[-1] // (3 * self.head_dim)
        new_qkv_shape = qkv_states.shape[:-1] + (num_heads, 3 * self.head_dim)
        qkv_states = qkv_states.view(new_qkv_shape)
        qkv_states = qkv_states.permute((0, 2, 1, 3))
        query_states, key_states, value_states = torch.chunk(qkv_states, 3, dim=-1)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # query_states = self.q_proj(hidden_states) * self.scaling
        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        #     value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)
        # else:
        #     # self_attention
        #     key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        #     value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        past_key_value = (key_states, value_states)

        proj_shape = (bsz * num_heads, -1, self.head_dim)
        query_states = (query_states * self.scaling).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * num_heads, tgt_len, src_len):
            raise ValueError(f"Attention weights should be of size {(bsz * num_heads, tgt_len, src_len)}, but is"
                             f" {attn_weights.size()}")

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}")
            attn_weights = attn_weights.view(bsz, num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(bsz * num_heads, tgt_len, src_len)
            dtype_attn_weights = attn_weights.dtype

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if dtype_attn_weights == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype_attn_weights)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (num_heads,):
                raise ValueError(f"Head mask for a single layer should be of size {(num_heads,)}, but is"
                                 f" {layer_head_mask.size()}")
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * num_heads, tgt_len, src_len)

        attn_probs = self.dropout(attn_weights)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * num_heads, tgt_len, self.head_dim):
            raise ValueError(f"`attn_output` should be of size {(bsz, num_heads, tgt_len, self.head_dim)}, but is"
                             f" {attn_output.size()}")

        attn_output = attn_output.view(bsz, num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output`
        # can be partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, past_key_value


class OPTDecoderLayer(nn.Module):

    def __init__(self,
                 hidden_size: int = 768,
                 ffn_dim: int = 3072,
                 num_attention_heads: int = 12,
                 attention_dropout: float = 0.0,
                 dropout: float = 0.1,
                 do_layer_norm_before: bool = True,
                 activation_function: Callable = nn.ReLU(),
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.embed_dim = hidden_size
        self.self_attn = OPTAttention(embed_dim=self.embed_dim,
                                      num_heads=num_attention_heads,
                                      dropout=attention_dropout)
        self.dropout = cube_nn.Dropout(dropout)
        self.do_layer_norm_before = do_layer_norm_before
        self.self_attn_layer_norm = cube_nn.LayerNorm(self.embed_dim)

        self.activation_fn = activation_function
        self.fc1 = cube_nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = cube_nn.Linear(ffn_dim, self.embed_dim)
        self.final_layer_norm = cube_nn.LayerNorm(self.embed_dim)
        self.gradient_checkpointing = gradient_checkpointing

    def _forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ):
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, present_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
    ):
        if self.gradient_checkpointing and self.training:
            hidden_states, _ = torch.utils.checkpoint.checkpoint(self._forward, hidden_states, attention_mask,
                                                                 layer_head_mask)
            return (hidden_states,)
        else:
            hidden_states, present_key_value = self._forward(hidden_states, attention_mask, layer_head_mask,
                                                             past_key_value)
            outputs = (hidden_states,)
            if use_cache:
                outputs += (present_key_value,)
            return outputs


class OPTLMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = cube_nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


class OPT(nn.Module):

    def __init__(self,
                 vocab_size: int = 50304,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 ffn_dim: int = 3072,
                 num_attention_heads: int = 12,
                 max_position_embeddings: int = 1024,
                 do_layer_norm_before: bool = True,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.0,
                 layerdrop: float = 0.0,
                 padding_idx: int = 1,
                 activation_function: Callable = nn.ReLU(),
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.dropout = dropout
        self.layerdrop = layerdrop
        self.padding_idx = padding_idx
        self.max_target_positions = max_position_embeddings
        self.vocab_size = vocab_size

        self.embed_tokens = cube_nn.Embedding(vocab_size,
                                              hidden_size,
                                              vocab_parallel=True,
                                              padding_idx=self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(max_position_embeddings, hidden_size)

        self.layers = nn.ModuleList([
            OPTDecoderLayer(
                hidden_size=hidden_size,
                ffn_dim=ffn_dim,
                num_attention_heads=num_attention_heads,
                attention_dropout=attention_dropout,
                dropout=dropout,
                do_layer_norm_before=do_layer_norm_before,
                activation_function=activation_function,
                gradient_checkpointing=gradient_checkpointing,
            ) for _ in range(num_hidden_layers)
        ])

        if do_layer_norm_before:
            self.final_layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.final_layer_norm = None

        self.lm_head = cube_nn.Classifier(hidden_size, vocab_size, weight=self.embed_tokens.weight, bias=False)

    def _make_causal_mask(self, input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
        mask_cond = torch.arange(mask.size(-1))
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    def _expand_mask(self, mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(input_shape,
                                                             inputs_embeds.dtype,
                                                             past_key_values_length=past_key_values_length).to(
                                                                 inputs_embeds.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = self._expand_mask(attention_mask, inputs_embeds.dtype,
                                                   tgt_len=input_shape[-1]).to(inputs_embeds.device)
            combined_attention_mask = (expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask +
                                       combined_attention_mask)

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        use_cache: Optional[bool] = None,
    ):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds,
                                                              past_key_values_length)

        # embed positions
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        hidden_states = inputs_embeds + pos_embeds

        # decoder layers
        next_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}.")

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                past_key_value=past_key_value,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_cache += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.lm_head(hidden_states)

        outputs = (hidden_states,)
        if use_cache:
            outputs += (next_cache,)
        return outputs


def opt_125m(max_position_embeddings=1024, checkpoint=False):
    model_kwargs = dict(
        max_position_embeddings=max_position_embeddings,
        hidden_size=768,
        ffn_dim=3072,
        num_hidden_layers=12,
        num_attention_heads=16,
        gradient_checkpointing=checkpoint,
    )
    return OPT(**model_kwargs)


def opt_350m(max_position_embeddings=1024, checkpoint=False):
    model_kwargs = dict(
        max_position_embeddings=max_position_embeddings,
        hidden_size=1024,
        ffn_dim=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        gradient_checkpointing=checkpoint,
    )
    return OPT(**model_kwargs)


def opt_1b(max_position_embeddings=1024, checkpoint=False):
    model_kwargs = dict(
        max_position_embeddings=max_position_embeddings,
        hidden_size=2048,
        ffn_dim=8192,
        num_hidden_layers=24,
        num_attention_heads=32,
        gradient_checkpointing=checkpoint,
    )
    return OPT(**model_kwargs)


def opt_3b(max_position_embeddings=1024, checkpoint=False):
    model_kwargs = dict(
        max_position_embeddings=max_position_embeddings,
        hidden_size=2560,
        ffn_dim=10240,
        num_hidden_layers=32,
        num_attention_heads=32,
        gradient_checkpointing=checkpoint,
    )
    return OPT(**model_kwargs)


def opt_6b(max_position_embeddings=1024, checkpoint=False):
    model_kwargs = dict(
        max_position_embeddings=max_position_embeddings,
        hidden_size=4096,
        ffn_dim=16384,
        num_hidden_layers=32,
        num_attention_heads=32,
        gradient_checkpointing=checkpoint,
    )
    return OPT(**model_kwargs)


def opt_13b(max_position_embeddings=1024, checkpoint=True):
    model_kwargs = dict(
        max_position_embeddings=max_position_embeddings,
        hidden_size=5120,
        ffn_dim=20480,
        num_hidden_layers=40,
        num_attention_heads=64,
        gradient_checkpointing=checkpoint,
    )
    return OPT(**model_kwargs)


def opt_30b(max_position_embeddings=1024, checkpoint=True):
    model_kwargs = dict(
        max_position_embeddings=max_position_embeddings,
        hidden_size=7168,
        ffn_dim=28672,
        num_hidden_layers=48,
        num_attention_heads=64,
        gradient_checkpointing=checkpoint,
    )
    return OPT(**model_kwargs)


def opt_66b(max_position_embeddings=1024, checkpoint=True):
    model_kwargs = dict(
        max_position_embeddings=max_position_embeddings,
        hidden_size=9216,
        ffn_dim=36864,
        num_hidden_layers=64,
        num_attention_heads=64,
        gradient_checkpointing=checkpoint,
    )
    return OPT(**model_kwargs)


def opt_175b(max_position_embeddings=1024, checkpoint=True):
    model_kwargs = dict(
        max_position_embeddings=max_position_embeddings,
        hidden_size=12288,
        ffn_dim=49152,
        num_hidden_layers=96,
        num_attention_heads=128,
        gradient_checkpointing=checkpoint,
    )
    return OPT(**model_kwargs)


def build_model(args):
    logger = get_logger()
    model_func = globals()[args.model_name]
    model = model_func(max_position_embeddings=args.seq_length, checkpoint=args.use_activation_checkpoint)
    logger.info("Model is built.")
    return model


def _tokenize(examples, tokenizer, seq_length):
    tokenizer.pad_token = tokenizer.unk_token
    examples = list(map(lambda x: x["text"], examples))
    batch = tokenizer(examples, padding="max_length", truncation=True, max_length=seq_length, return_tensors="pt")
    batch["labels"] = batch["input_ids"].clone()
    return batch


def build_data(args):
    logger = get_logger()
    dataset = load_from_disk(args.dataset_path)
    dataset = dataset['train'].train_test_split(train_size=0.9, test_size=0.1, shuffle=False)
    logger.info(f"Loaded dataset:\n{dataset}")
    tokenizer = GPT2Tokenizer(vocab_file=args.tokenizer_path + "/vocab.json",
                              merges_file=args.tokenizer_path + "/merges.txt")

    train_data = get_dataloader(
        dataset=dataset["train"],
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,
        collate_fn=partial(_tokenize, tokenizer=tokenizer, seq_length=args.seq_length),
        num_workers=4,
        pin_memory=True,
    )
    test_data = get_dataloader(
        dataset=dataset["test"],
        batch_size=args.batch_size,
        collate_fn=partial(_tokenize, tokenizer=tokenizer, seq_length=args.seq_length),
        num_workers=4,
        pin_memory=True,
    )
    logger.info("Train and test data are built.")

    return train_data, test_data


def build_criterion():
    logger = get_logger()
    criterion = OPTLMLoss()
    metric = cube_nn.Perplexity()
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
    warmup_steps = n_steps * args.warmup_epochs
    lr_scheduler = get_linear_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=warmup_steps,
                                                   num_training_steps=max_steps)
    logger.info("Learning rate scheduler is built.")
    return lr_scheduler


def build_opt(args):
    model = build_model(args)

    train_data, test_data = build_data(args)

    criterion, metric = build_criterion()

    optimizer = build_optimizer(args, model.parameters())
    n_steps = len(train_data)
    if args.steps_per_epoch is not None and args.steps_per_epoch < n_steps:
        n_steps = args.steps_per_epoch
    lr_scheduler = build_scheduler(args, n_steps, optimizer)

    return model, train_data, test_data, criterion, metric, optimizer, lr_scheduler
