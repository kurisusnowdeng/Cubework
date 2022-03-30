import argparse
import time

import cubework
import torch
from cubework.arguments import get_args
from cubework.distributed import ParallelManager as pm
from cubework.distributed.collective import all_reduce
from cubework.utils import (
    CommProfiler,
    MemoryTracker,
    calc_model_size,
    calc_tflops,
    clip_grad_norm,
    get_current_device,
    get_logger,
    write_logger_to_file,
)
from tqdm import tqdm

from gpt2 import build_gpt2
from vit import build_vit

_builder = {
    "gpt2": build_gpt2,
    "vit": build_vit,
}


logger = None
mem_tracker = None
comm_profiler = None

model = None
train_data = None
test_data = None
criterion = None
metric = None
optimizer = None
scaler = None
lr_scheduler = None

numel = None


def _data_parallel_sum(tensor):
    out = tensor
    if pm.DATA.world_size > 1:
        out = all_reduce(out, pm.DATA)
    return out


def _data_parallel_mean(tensor):
    out = tensor
    if pm.DATA.world_size > 1:
        out = all_reduce(out, pm.DATA) / pm.DATA.world_size
    return out


def _move_to_cuda(x):
    if isinstance(x, dict):
        return {k: _move_to_cuda(v) for k, v in x.items()}
    elif isinstance(x, (tuple, list)):
        return type(x)(_move_to_cuda(v) for v in x)
    else:
        return x.to(get_current_device())


def _train(epoch, args):
    model.train()

    num_steps = len(train_data)
    if args.steps_per_epoch is not None and args.steps_per_epoch < num_steps:
        num_steps = args.steps_per_epoch
    progress = range(num_steps)

    if pm.GLOBAL.rank == 0:
        progress = tqdm(progress, desc=f"[Epoch {epoch} / Train]")

    total_loss = torch.zeros(()).to(torch.float).to(get_current_device())
    total_time = 0.0
    total_steps = 0
    total_samples = torch.zeros(()).to(torch.int).to(get_current_device())
    total_tokens = torch.zeros(()).to(torch.int).to(get_current_device())

    data_iter = iter(train_data)

    if comm_profiler is not None:
        comm_profiler.reset()
        comm_profiler.start()

    if mem_tracker is not None:
        mem_tracker.start()

    for i in progress:
        fwd_start = time.time()

        batch = _move_to_cuda(next(data_iter))

        labels = batch.pop("labels")

        if args.use_mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
        else:
            outputs = model(**batch)

        batch_size = outputs.size(0)
        batch_tokens = outputs.size(0) * outputs.size(1)

        loss = criterion(outputs, labels)
        total_loss += loss

        fwd_end = time.time()

        bwd_start = time.time()

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % args.gradient_accumulation or i + 1 == num_steps:
            if scaler is not None:
                if args.gradient_clipping > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm(model.parameters(), args.gradient_clipping)
                scaler.step(optimizer)
                scaler.update()

            else:
                if args.gradient_clipping > 0:
                    clip_grad_norm(model.parameters(), args.gradient_clipping)
                optimizer.step()

        lr_scheduler.step()

        bwd_end = time.time()

        total_steps += 1
        total_samples += batch_size
        total_tokens += batch_tokens

        fwd_time = fwd_end - fwd_start
        bwd_time = bwd_end - bwd_start
        batch_time = fwd_time + bwd_time
        total_time += batch_time

        if pm.GLOBAL.rank == 0:
            batch_tflops = calc_tflops(
                numel,
                batch_tokens * pm.DATA.world_size,
                batch_time,
                with_backward=True,
                checkpoint=args.use_activation_checkpoint,
            )
            progress.set_postfix(
                loss=loss.item(),
                lr=lr_scheduler.get_last_lr()[0],
                time_forward=fwd_time,
                time_backward=bwd_time,
                throughput=batch_size * pm.DATA.world_size / (batch_time + 1e-12),
                tflops=batch_tflops,
            )

    if mem_tracker is not None:
        peak_mem = mem_tracker.stop()

    if comm_profiler is not None:
        _, comm_vol, comm_time = comm_profiler.stop()

    total_loss = _data_parallel_mean(total_loss)
    total_samples = _data_parallel_sum(total_samples)
    total_tokens = _data_parallel_sum(total_tokens)

    msg = f"[Epoch {epoch} / Train]: Loss = {total_loss.item() / num_steps:.3f}"
    msg += f" | Throughput = {total_samples.item() / (total_time + 1e-12):.3f} samples/sec"
    tflops = calc_tflops(
        numel, total_tokens.item(), total_time, with_backward=True, checkpoint=args.use_activation_checkpoint
    )
    msg += f" | TFLOPS = {tflops:.3f}"
    if mem_tracker is not None:
        msg += f" | Peak memory = {peak_mem / 1024:.3f} GB"
    if comm_profiler is not None:
        msg += (
            f"\n[Epoch {epoch} / Train]: Communication time = {comm_time:.3f} s, "
            + f"ratio = {comm_time * 100 / (total_time + 1e-12):.3f} %, "
            + f"avg bandwidth = {(comm_vol / 1024**2) / (comm_time + 1e-12):.3f} MB/s"
        )
    logger.info(msg)


def _test(epoch, args):
    model.eval()

    num_steps = len(test_data)
    if args.steps_per_epoch is not None and args.steps_per_epoch < num_steps:
        num_steps = args.steps_per_epoch
    progress = range(num_steps)

    if pm.GLOBAL.rank == 0:
        progress = tqdm(progress, desc=f"[Epoch {epoch} / Test]")

    total_loss = torch.zeros(()).to(torch.float).to(get_current_device())
    total_time = 0.0
    total_steps = 0
    total_samples = torch.zeros(()).to(torch.int).to(get_current_device())
    total_tokens = torch.zeros(()).to(torch.int).to(get_current_device())
    metric.reset()

    data_iter = iter(test_data)

    if comm_profiler is not None:
        comm_profiler.reset()
        comm_profiler.start()

    if mem_tracker is not None:
        mem_tracker.start()

    with torch.no_grad():
        for _ in progress:
            batch_start = time.time()

            batch = _move_to_cuda(next(data_iter))

            labels = batch.pop("labels")

            if args.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
            else:
                outputs = model(**batch)

            batch_size = outputs.size(0)
            batch_tokens = outputs.size(0) * outputs.size(1)

            loss = criterion(outputs, labels)
            eval_res = metric(outputs, labels, loss)
            total_loss += loss

            batch_end = time.time()

            total_steps += 1
            total_samples += batch_size
            total_tokens += batch_tokens

            batch_time = batch_end - batch_start
            total_time += batch_time

            if pm.GLOBAL.rank == 0:
                batch_tflops = calc_tflops(
                    numel,
                    batch_tokens * pm.DATA.world_size,
                    batch_time,
                    with_backward=False,
                    checkpoint=False,
                )
                metrics = dict(
                    loss=loss.item(),
                    step_time=batch_time,
                    throughput=batch_size * pm.DATA.world_size / (batch_time + 1e-12),
                    tflops=batch_tflops,
                )
                metrics[metric.name.lower()] = eval_res
                progress.set_postfix(**metrics)

    if mem_tracker is not None:
        peak_mem = mem_tracker.stop()

    if comm_profiler is not None:
        _, comm_vol, comm_time = comm_profiler.stop()

    total_loss = _data_parallel_mean(total_loss)
    total_samples = _data_parallel_sum(total_samples)
    total_tokens = _data_parallel_sum(total_tokens)

    msg = f"[Epoch {epoch} / Train]: Loss = {total_loss.item() / num_steps:.3f}"
    msg += f" | {metric.name} = {metric.value().item():.5f}"
    msg += f" | Throughput = {total_samples.item() / (total_time + 1e-12):.3f} samples/sec"
    tflops = calc_tflops(
        numel, total_tokens.item(), total_time, with_backward=True, checkpoint=args.use_activation_checkpoint
    )
    msg += f" | TFLOPS = {tflops:.3f}"
    if mem_tracker is not None:
        msg += f" | Peak memory = {peak_mem / 1024:.3f} GB"
    if comm_profiler is not None:
        msg += (
            f"\n[Epoch {epoch} / Train]: Communication time = {comm_time:.3f} s, "
            + f"ratio = {comm_time * 100 / (total_time + 1e-12):.3f} %, "
            + f"avg bandwidth = {(comm_vol / 1024**2) / (comm_time + 1e-12):.3f} MB/s"
        )
    logger.info(msg)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", "--m", type=str)
    parser.add_argument("--dataset_path", "--data", type=str)
    parser.add_argument("--tokenizer_path", "--token", type=str)

    parser.add_argument("--batch_size", "--bs", type=int)
    parser.add_argument("--num_epochs", "--n_epoch", type=int)
    parser.add_argument("--warmup_epochs", "--n_warm", type=int, default=0)
    parser.add_argument("--steps_per_epoch", "--n_step", type=int)
    parser.add_argument("--learning_rate", "--lr", type=float)
    parser.add_argument("--weight_decay", "--decay", type=float)

    parser.add_argument("--use_activation_checkpoint", "--ckpt", action="store_true", default=False)

    parser.add_argument("--gradient_clipping", "--clip", type=float, default=0.0)

    parser.add_argument("--gradient_accumulation", "--ac", type=int, default=1)

    parser.add_argument("--use_mixed_precision", "--amp", action="store_true", default=False)
    parser.add_argument("--fp16_initial_scale", type=float, default=2**15)
    parser.add_argument("--fp16_growth_factor", type=float, default=2.0)
    parser.add_argument("--fp16_backoff_factor", type=float, default=0.5)
    parser.add_argument("--fp16_growth_interval", type=int, default=1000)

    parser.add_argument("--use_mem_tracker", "--prof_mem", action="store_true", default=False)
    parser.add_argument("--use_comm_profiler", "--prof_comm", action="store_true", default=False)

    parser.add_argument("--log_file", type=str)
    return parser


def train():
    parser = get_parser()
    cubework.initialize_distributed(parser)
    args = get_args()

    logger = get_logger()
    if args.log_file is not None:
        write_logger_to_file(logger)

    model_type = args.model_name.split("_")[0]
    assert model_type in ["gpt2", "vit"], f"No support for {args.model_name}."

    global model, train_data, test_data, criterion, metric, optimizer, lr_scheduler
    model, train_data, test_data, criterion, metric, optimizer, lr_scheduler = _builder[model_type](args)

    global scaler
    if args.use_mixed_precision:
        scaler = torch.cuda.amp.GradScaler(
            enabled=True,
            initial_scale=args.fp16_initial_scale,
            growth_factor=args.fp16_growth_factor,
            backoff_factor=args.fp16_backoff_factor,
            growth_interval=args.fp16_growth_interval,
        )

    global mem_tracker
    if args.use_mem_monitor:
        mem_tracker = MemoryTracker(args.log_file)

    global comm_profiler
    if args.use_comm_profiler:
        comm_profiler = CommProfiler()

    global numel
    numel = calc_model_size(model)
    if numel < 1e9:
        msg = f"{numel / 1e6:.3f} M"
    else:
        msg = f"{numel / 1e9:.3f} B"
    logger.info(f"Model is built (parameter size = {msg}).")

    logger.info("Benchmark start.")

    for epoch in range(args.num_epochs):
        _train(epoch, args)
        _test(epoch, args)

    logger.info("Benchmark complete.")


if __name__ == "__main__":
    train()
