"""
Testing the speed of different models in FP16 (on CUDA) by default
"""
import os
import torch
import time
from model import get_model
from argparse import Namespace as _Namespace

torch.autograd.set_grad_enabled(False)

T0 = 5  # warmup time
T1 = 10  # benchmark time

def compute_throughput_cpu(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    # CPU always uses FP32
    start = time.time()
    while time.time() - start < T0:
        model(inputs)

    timing = []
    while sum(timing) < T1:
        start = time.time()
        model(inputs)
        timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)

def compute_throughput_cuda(name, model, device, batch_size, resolution=224):
    # Create FP16 input
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device, dtype=torch.float16)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Warmup
    start = time.time()
    while time.time() - start < T0:
        _ = model(inputs)
    torch.cuda.synchronize()

    # Benchmark
    timing = []
    while sum(timing) < T1:
        start = time.time()
        _ = model(inputs)
        torch.cuda.synchronize()
        timing.append(time.time() - start)

    timing = torch.as_tensor(timing, dtype=torch.float32)
    print(name, device, batch_size / timing.mean().item(),
          'images/s @ batch size', batch_size)

# Main loop
for device in ['cuda:0', 'cpu']:
    if 'cuda' in device and not torch.cuda.is_available():
        print("No CUDA available, skipping.")
        continue

    if device == 'cpu':
        os.system('echo -n "nb processors "; '
                  'cat /proc/cpuinfo | grep ^processor | wc -l; '
                  'cat /proc/cpuinfo | grep ^"model name" | tail -1')
        print('Using 1 CPU thread')
        torch.set_num_threads(1)
        compute_throughput = compute_throughput_cpu
    else:
        print("Device:", torch.cuda.get_device_name(torch.cuda.current_device()))
        compute_throughput = compute_throughput_cuda

    for n, batch_size0, resolution in [
        ('StarNet_MHSA_T2_DTW', 2096, 192),
        ('StarNet_MHSA_T4_DTW', 2048, 192),
    ]:

        if device == 'cpu':
            batch_size = 16
        else:
            batch_size = batch_size0
            torch.cuda.empty_cache()

        # Create model
        model_cfg = _Namespace()
        model_cfg.name = n
        model_cfg.model_kwargs = dict(pretrained=False, checkpoint_path='', ema=False, strict=True, num_classes=1000)
        model = get_model(model_cfg)

        # Move to device and convert to FP16 if on CUDA
        model.to(device)
        if device != 'cpu':
            model = model.half()  # <-- Convert model to FP16
        model.eval()

        # Prepare FP16 input for tracing (if CUDA)
        dummy_input = torch.randn(
            batch_size, 3, resolution, resolution,
            device=device,
            dtype=torch.float16 if device != 'cpu' else torch.float32
        )

        # Trace the model
        model = torch.jit.trace(model, dummy_input)

        # Benchmark
        compute_throughput(n, model, device, batch_size, resolution=resolution)