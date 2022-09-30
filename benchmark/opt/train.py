import argparse
import time

import cubework
import torch
from cubework.distributed import ParallelManager as pm
from cubework.distributed.collective import all_reduce
from cubework.utils import (CommProfiler, calc_model_size, calc_tflops, clip_grad_norm, get_current_device, get_logger,
                            write_logger_to_file)
from fairscale.optim.grad_scaler import GradScaler, ShardedGradScaler
from tqdm import tqdm

from opt import build_opt

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


def move_to_cuda(x):
    if isinstance(x, dict):
        return {k: move_to_cuda(v) for k, v in x.items()}
    elif isinstance(x, (tuple, list)):
        return type(x)(move_to_cuda(v) for v in x)
    else:
        return x.to(get_current_device())


def _train(epoch, args):
    logger = get_logger()

    model.train()

    num_steps = len(train_data)
    accum_size = args.global_batch_size // (args.micro_batch_size * pm.DATA.world_size)
    num_steps = num_steps // accum_size
    if args.steps_per_epoch is not None:
        num_steps = min(args.steps_per_epoch, num_steps)
    progress = range(num_steps)
    if pm.GLOBAL.rank == 0:
        progress = tqdm(progress, desc=f"[Epoch {epoch} / Train]")

    train_loss = 0.
    num_tokens = 0
    num_samples = 0
    data_time = 0.
    fwd_time = 0.
    bwd_time = 0.
    opt_time = 0.

    data_iter = iter(train_data)

    torch.cuda.reset_peak_memory_stats()

    if comm_profiler is not None:
        comm_profiler.reset()
        comm_profiler.start()

    for i in progress:
        for _ in range(accum_size):
            data_start = time.time()
            batch = move_to_cuda(next(data_iter))
            labels = batch.pop("labels")
            num_tokens += labels.numel()
            num_samples += labels.size(0)
            data_end = time.time()

            fwd_start = time.time()
            if args.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
            else:
                outputs = model(**batch)
            loss = criterion(outputs[0], labels)
            train_loss += loss.item()
            torch.cuda.synchronize()
            fwd_end = time.time()

            bwd_start = time.time()
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            torch.cuda.synchronize()
            bwd_end = time.time()

            data_time += data_end - data_start
            fwd_time += fwd_end - fwd_start
            bwd_time += bwd_end - bwd_start

        opt_start = time.time()
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
        optimizer.zero_grad()
        torch.cuda.synchronize()
        opt_end = time.time()

        opt_time += opt_end - opt_start

        if pm.GLOBAL.rank == 0:
            avg_time = (data_time + fwd_time + bwd_time + opt_time) / (i + 1)
            avg_samples = num_samples * pm.DATA.world_size / (i + 1)
            avg_tokens = num_tokens * pm.DATA.world_size / (i + 1)
            batch_tflops = calc_tflops(
                numel,
                avg_tokens,
                avg_time,
                with_backward=True,
                use_checkpoint=args.use_activation_checkpoint,
            )
            progress.set_postfix(
                loss=loss.item(),
                lr=lr_scheduler.get_last_lr()[0],
                time_dataloader=data_time / (i + 1),
                time_forward=fwd_time / (i + 1),
                time_backward=bwd_time / (i + 1),
                time_optimizer=opt_time / (i + 1),
                throughput=avg_samples / avg_time,
                tflops=batch_tflops,
            )

    used_time = data_time + fwd_time + bwd_time + opt_time

    if comm_profiler is not None:
        comm_cnt, comm_vol, comm_time = comm_profiler.stop()

    train_loss, num_samples, num_tokens = aggregate_ddp_results(train_loss, num_samples, num_tokens)

    msg = f"[Epoch {epoch} / Train]: Loss = {train_loss / (pm.DATA.world_size * num_steps * accum_size):.3f}"
    msg += f" | Step time = {used_time / num_steps:.3f} s"
    msg += f" | Throughput = {num_samples / used_time:.3f} samples/sec, {num_tokens / used_time:.3f} tokens/sec"
    tflops = calc_tflops(numel,
                         num_tokens,
                         used_time,
                         with_backward=True,
                         use_checkpoint=args.use_activation_checkpoint)
    msg += f" | TFLOPS = {tflops:.3f}"

    torch.cuda.empty_cache()
    peak_mem = torch.cuda.max_memory_allocated()
    reserved_mem = torch.cuda.max_memory_reserved()

    msg += f"\n[Epoch {epoch} / Train]: Peak memory = {peak_mem / 1024**3:.3f} GB"
    msg += f" | Reserved memory = {reserved_mem / 1024**3:.3f} GB"
    state_mem = torch.cuda.memory_allocated() - model_mem
    msg += f" | Gradients & optimizer states memory = {state_mem / 1024**3:.3f} GB."
    activation_mem = peak_mem - state_mem - model_mem
    msg += f" | Activation memory = {activation_mem / 1024**3:.3f} GB."

    if comm_profiler is not None:
        msg += f"\n[Epoch {epoch} / Train]: Communication total time = {comm_time:.3f} s, # ops = {comm_cnt:.5g}"
        if comm_time > 0:
            msg += (f"| Comm time per step = {comm_time / num_steps:.3f} s"
                    f"| Ratio = {comm_time * 100 / used_time:.3f} %"
                    f"| Avg bandwidth = {comm_vol / (comm_time * 1024**3):.3f} GB/s")
    logger.info(msg)


def _test(epoch, args):
    logger = get_logger()

    model.eval()

    num_steps = len(train_data)
    if args.steps_per_epoch is not None:
        num_steps = min(args.steps_per_epoch, num_steps)
    progress = range(num_steps)
    if pm.GLOBAL.rank == 0:
        progress = tqdm(progress, desc=f"[Epoch {epoch} / Train]")

    train_loss = 0.
    num_tokens = 0
    num_samples = 0
    data_time = 0.
    fwd_time = 0.
    metric.reset()

    data_iter = iter(test_data)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if comm_profiler is not None:
        comm_profiler.reset()
        comm_profiler.start()

    epoch_start = time.time()

    with torch.no_grad():
        for i in progress:
            data_start = time.time()
            batch = move_to_cuda(next(data_iter))
            labels = batch.pop("labels")
            num_tokens += labels.numel()
            num_samples += labels.size(0)
            data_end = time.time()

            fwd_start = time.time()
            if args.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
            else:
                outputs = model(**batch)
            loss = criterion(outputs[0], labels)
            train_loss += loss.item()
            eval_res = metric(outputs[0], labels, loss)
            torch.cuda.synchronize()
            fwd_end = time.time()

            data_time += data_end - data_start
            fwd_time += fwd_end - fwd_start

            if pm.GLOBAL.rank == 0:
                avg_time = (time.time() - epoch_start) / (i + 1)
                avg_tokens = num_tokens * pm.DATA.world_size / (i + 1)
                batch_tflops = calc_tflops(
                    numel,
                    avg_tokens,
                    avg_time,
                    with_backward=False,
                    use_checkpoint=False,
                )
                outputs = dict(
                    loss=loss.item(),
                    lr=lr_scheduler.get_last_lr()[0],
                    time_dataloader=data_time / (i + 1),
                    time_forward=fwd_time / (i + 1),
                    throughput=avg_tokens / avg_time,
                    tflops=batch_tflops,
                )
                outputs[metric.name.lower()] = eval_res.item()
                progress.set_postfix(**outputs)

    epoch_end = time.time()
    used_time = epoch_end - epoch_start
    peak_mem = torch.cuda.max_memory_allocated()

    if comm_profiler is not None:
        comm_cnt, comm_vol, comm_time = comm_profiler.stop()

    train_loss, num_tokens = aggregate_ddp_results(train_loss, num_tokens)

    msg = f"[Epoch {epoch} / Train]: Loss = {train_loss / (pm.DATA.world_size * num_steps):.3f}"
    msg += f" | {metric.name} = {metric.to_str()}"
    msg += f" | Step time = {used_time / num_steps:.3f} s"
    msg += f" | Throughput = {num_samples / used_time:.3f} samples/sec, {num_tokens / used_time:.3f} tokens/sec"
    tflops = calc_tflops(numel, num_tokens, used_time, with_backward=False, use_checkpoint=False)
    msg += f" | TFLOPS = {tflops:.3f}"

    msg += f"\n[Epoch {epoch} / Train]: Peak memory = {peak_mem / 1024**3:.3f} GB"
    activation_mem = peak_mem - model_mem
    msg += f" | Activation memory = {activation_mem / 1024**3:.3f} GB."

    if comm_profiler is not None:
        msg += f"\n[Epoch {epoch} / Train]: Communication total time = {comm_time:.3f} s, # ops = {comm_cnt:.5g}"
        if comm_time > 0:
            msg += (f", avg step time = {comm_time / num_steps:.3f} s"
                    f", ratio = {comm_time * 100 / used_time:.3f} %"
                    f", avg bandwidth = {comm_vol / (comm_time * 1024**3):.3f} GB/s")
    logger.info(msg)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", "-m", type=str)
    parser.add_argument("--seq_length", "-s", type=int)
    parser.add_argument("--dataset_path", "--data", type=str)
    parser.add_argument("--tokenizer_path", "--tok", type=str)

    parser.add_argument("--global_batch_size", "--bs", type=int)
    parser.add_argument("--micro_batch_size", "--micro_bs", type=int)

    parser.add_argument("--num_epochs", "--n_epoch", type=int)
    parser.add_argument("--warmup_steps", "--n_warm", type=int, default=0)
    parser.add_argument("--steps_per_epoch", "--n_step", type=int)
    parser.add_argument("--learning_rate", "--lr", type=float)
    parser.add_argument("--weight_decay", "--decay", type=float, default=0.01)

    parser.add_argument("--do_validation", "--eval", action="store_true", default=False)
    parser.add_argument("--validation_interval", "--n_eval", type=int, default=1)

    parser.add_argument("--use_fully_sharded_data_parallel", "--fsdp", action="store_true", default=False)

    parser.add_argument("--use_activation_checkpoint", "--ckpt", action="store_true", default=False)

    parser.add_argument("--gradient_clipping", "--clip", type=float, default=0.0)

    parser.add_argument("--use_mixed_precision", "--amp", action="store_true", default=False)
    parser.add_argument("--use_amp_opt_level_3", "-O3", action="store_true", default=False)
    parser.add_argument("--fp16_initial_scale", type=float, default=2**15)
    parser.add_argument("--fp16_growth_factor", type=float, default=2.0)
    parser.add_argument("--fp16_backoff_factor", type=float, default=0.5)
    parser.add_argument("--fp16_growth_interval", type=int, default=1000)

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

    global scaler
    if args.use_mixed_precision:
        scaler_cls = torch.cuda.amp.GradScaler
        if args.use_amp_opt_level_3:
            scaler_cls = GradScaler
        elif args.use_fully_sharded_data_parallel:
            scaler_cls = ShardedGradScaler
        scaler = scaler_cls(
            enabled=True,
            init_scale=args.fp16_initial_scale,
            growth_factor=args.fp16_growth_factor,
            backoff_factor=args.fp16_backoff_factor,
            growth_interval=args.fp16_growth_interval,
        )

    global comm_profiler
    if args.use_communication_profiler:
        comm_profiler = CommProfiler()

    global numel
    numel, _ = calc_model_size(model)
    if args.use_fully_sharded_data_parallel:
        numel *= pm.DATA.world_size
        if pm.TENSOR.is_initialized():
            numel *= pm.TENSOR.world_size
    if numel < 1e9:
        msg = f"{numel / 1e6:.3f} M"
    else:
        msg = f"{numel / 1e9:.3f} B"
    global model_mem
    torch.cuda.empty_cache()
    model_mem = torch.cuda.memory_allocated()
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
