import argparse
import time

import cubework
import torch
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
from cubework.module import synchronize
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from opt import build_opt

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
model_mem = None


def aggregate_ddp_results(*vals):
    tensor = torch.as_tensor(vals, device=get_current_device())
    all_reduce(tensor, pm.DATA)
    return tuple(tensor.tolist())


def _move_to_cuda(x):
    if isinstance(x, dict):
        return {k: _move_to_cuda(v) for k, v in x.items()}
    elif isinstance(x, (tuple, list)):
        return type(x)(_move_to_cuda(v) for v in x)
    else:
        return x.to(get_current_device())


def _train(epoch, args):
    logger = get_logger()

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

    past_key_values = None
    for i in progress:
        fwd_start = time.time()

        batch = _move_to_cuda(next(data_iter))

        labels = batch.pop("labels")
        if args.use_cache:
            batch['past_key_values'] = past_key_values

        if args.use_mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
        else:
            outputs = model(**batch)
        if args.use_cache:
            past_key_values = outputs[1]
        outputs = outputs[0]

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

        synchronize(model.parameters())

        if (i + 1) % args.gradient_accumulation or i + 1 == num_steps:
            if scaler is not None:
                if args.gradient_clipping > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm(model.parameters(), args.gradient_clipping)
                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()
                # skip stepping lr scheduler if overflow
                if not scale > scaler.get_scale():
                    lr_scheduler.step()
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
                throughput=batch_size * pm.DATA.world_size / batch_time,
                tflops=batch_tflops,
            )

    torch.cuda.synchronize()

    if mem_tracker is not None:
        peak_mem = mem_tracker.stop()

    if comm_profiler is not None:
        comm_cnt, comm_vol, comm_time = comm_profiler.stop()

    total_loss, total_samples, total_tokens = aggregate_ddp_results(total_loss, total_samples, total_tokens)

    msg = f"[Epoch {epoch} / Train]: Loss = {total_loss / total_steps:.3f}"
    msg += f" | Step time = {total_time / total_steps:.3f} s"
    msg += f" | Throughput = {total_samples / total_time:.3f} samples/sec"
    tflops = calc_tflops(numel, total_tokens, total_time, with_backward=True, checkpoint=args.use_activation_checkpoint)
    msg += f" | TFLOPS = {tflops:.3f}"

    if mem_tracker is not None:
        msg += f"\n[Epoch {epoch} / Train]: Peak memory = {peak_mem / 1024:.3f} GB"
        state_mem = torch.cuda.memory_allocated() - model_mem
        msg += f" | Gradients & optimizer states memory = {state_mem / 1024**3:.3f} GB."
        activation_mem = peak_mem - state_mem - model_mem
        msg += f" | Activation memory = {activation_mem / 1024**3:.3f} GB."

    if comm_profiler is not None:
        msg += f"\n[Epoch {epoch} / Train]: Communication total time = {comm_time:.3f} s, # ops = {comm_cnt:.5g}"
        if comm_time > 0:
            msg += (f", avg step time = {comm_time / total_steps:.3f} s" +
                    f", ratio = {comm_time * 100 / total_time:.3f} %" +
                    f", avg bandwidth = {comm_vol / (comm_time * 1024**3):.3f} GB/s")
    logger.info(msg)


def _test(epoch, args):
    logger = get_logger()

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

    past_key_values = None
    with torch.no_grad():
        for _ in progress:
            batch_start = time.time()

            batch = _move_to_cuda(next(data_iter))

            labels = batch.pop("labels")
            if args.use_cache:
                batch['past_key_values'] = past_key_values

            if args.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
            else:
                outputs = model(**batch)
            if args.use_cache:
                past_key_values = outputs[1]
            outputs = outputs[0]

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
                    throughput=batch_size * pm.DATA.world_size / batch_time,
                    tflops=batch_tflops,
                )
                metrics[metric.name.lower()] = eval_res.item()
                progress.set_postfix(**metrics)

    torch.cuda.synchronize()

    if mem_tracker is not None:
        peak_mem = mem_tracker.stop()

    if comm_profiler is not None:
        _, comm_vol, comm_time = comm_profiler.stop()

    total_loss, total_samples, total_tokens = aggregate_ddp_results(total_loss, total_samples, total_tokens)

    msg = f"[Epoch {epoch} / Test]: Loss = {total_loss / total_steps:.3f}"
    msg += f" | {metric.name} = {metric.to_str()}"
    msg += f" | Step time = {total_time / total_steps:.3f} s"
    msg += f" | Throughput = {total_samples / total_time:.3f} samples/sec"
    tflops = calc_tflops(numel, total_tokens, total_time, with_backward=True, checkpoint=args.use_activation_checkpoint)
    msg += f" | TFLOPS = {tflops:.3f}"

    if mem_tracker is not None:
        msg += f" | Peak memory = {peak_mem / 1024:.3f} GB"

    if comm_profiler is not None:
        msg += f"\n[Epoch {epoch} / Test]: Communication total time = {comm_time:.3f} s"
        if comm_time > 0:
            msg += (f", avg step time = {comm_time / total_steps:.3f} s" +
                    f", ratio = {comm_time * 100 / total_time:.3f} %" +
                    f", avg bandwidth = {comm_vol / (comm_time * 1024**3):.3f} GB/s")

    logger.info(msg)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", "-m", type=str)
    parser.add_argument("--seq_length", "-s", type=int)
    parser.add_argument("--dataset_path", "--data", type=str)
    parser.add_argument("--tokenizer_path", "--tok", type=str)

    parser.add_argument("--batch_size", "--bs", type=int)
    parser.add_argument("--num_epochs", "--n_epoch", type=int)
    parser.add_argument("--warmup_epochs", "--n_warm", type=int, default=0)
    parser.add_argument("--steps_per_epoch", "--n_step", type=int)
    parser.add_argument("--learning_rate", "--lr", type=float)
    parser.add_argument("--weight_decay", "--decay", type=float, default=0.01)

    parser.add_argument("--do_validation", "--eval", action="store_true", default=False)
    parser.add_argument("--validation_interval", "--n_eval", type=int, default=1)

    parser.add_argument("--use_activation_checkpoint", "--ckpt", action="store_true", default=False)
    parser.add_argument("--use_cache", "--cache", action="store_true", default=False)

    parser.add_argument("--gradient_clipping", "--clip", type=float, default=0.0)

    parser.add_argument("--gradient_accumulation", "--ac", type=int, default=1)

    parser.add_argument("--use_mixed_precision", "--amp", action="store_true", default=False)
    parser.add_argument("--fp16_initial_scale", type=float, default=2**15)
    parser.add_argument("--fp16_growth_factor", type=float, default=2.0)
    parser.add_argument("--fp16_backoff_factor", type=float, default=0.5)
    parser.add_argument("--fp16_growth_interval", type=int, default=1000)

    parser.add_argument("--use_memory_tracker", "--prof_mem", action="store_true", default=False)
    parser.add_argument("--use_communication_profiler", "--prof_comm", action="store_true", default=False)

    parser.add_argument("--log_file", type=str)
    return parser


def train():
    parser = get_parser()
    cubework.initialize_distributed(parser)
    args = cubework.get_args()

    logger = get_logger()
    if args.log_file is not None:
        write_logger_to_file(args.log_file)

    global model, train_data, test_data, criterion, metric, optimizer, lr_scheduler
    model, train_data, test_data, criterion, metric, optimizer, lr_scheduler = build_opt(args)

    if pm.DATA.world_size > 1:
        model = DDP(model, process_group=pm.DATA.group)

    global scaler
    if args.use_mixed_precision:
        scaler = torch.cuda.amp.GradScaler(
            enabled=True,
            init_scale=args.fp16_initial_scale,
            growth_factor=args.fp16_growth_factor,
            backoff_factor=args.fp16_backoff_factor,
            growth_interval=args.fp16_growth_interval,
        )

    global mem_tracker
    if args.use_memory_tracker:
        mem_tracker = MemoryTracker(args.log_file)

    global comm_profiler
    if args.use_communication_profiler:
        comm_profiler = CommProfiler()

    global numel
    numel, _ = calc_model_size(model)
    if numel < 1e9:
        msg = f"{numel / 1e6:.3f} M"
    else:
        msg = f"{numel / 1e9:.3f} B"
    global model_mem
    model_mem = torch.cuda.max_memory_allocated()
    logger.info(f"Parameter size = {msg} | Model memory = {model_mem / 1024**3:.3f} GB.")

    logger.info("Benchmark start.")

    for epoch in range(args.num_epochs):
        _train(epoch, args)
        if args.do_validation:
            if (epoch + 1) % args.validation_interval == 0 or epoch + 1 == args.num_epochs:
                _test(epoch, args)

    logger.info("Benchmark complete.")
    cubework.destroy_distributed()


if __name__ == "__main__":
    train()
