import torch
import time
import argparse
from thop import profile, clever_format
import torch.cuda


from core.backbone import Backbone_Event 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corr_levels', default=4, type=int)
    parser.add_argument('--corr_radius', default=4, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--alternate_corr', action='store_true')
    return parser.parse_args([])

def benchmark():
    args = get_args()
    device = torch.device('cuda:0')
    
    # input Tensor: Batch=1, H=600, W=960
    dummy_lidar = torch.randn(1, 1, 600, 960, device=device)
    dummy_event = torch.randn(1, 5, 600, 960, device=device)

    # --------------------------------------------------------
    # Phase 1: Independent testing of Params and MACs
    # --------------------------------------------------------
    print("--- phase 1:  DGLER Params & FLOPs ---")
    model_for_thop = Backbone_Event(args).to(device)
    model_for_thop.eval()
    

    macs, params = profile(model_for_thop, inputs=(dummy_lidar, dummy_event, 12, None, True), verbose=False)
    macs_fmt, params_fmt = clever_format([macs, params], "%.3f")
    print(f"Total Params: {params_fmt} (Exact: {int(params)})")
    print(f"Total MACs (FLOPs): {macs_fmt}")
    
    # Completely destroy the contaminated model and clear the video memory.
    del model_for_thop
    torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Phase 2: Testing of Clean Model Runtime (Latency) and Peak Memory
    # --------------------------------------------------------
    print("\n--- phase 2:  Runtime & Memory ---")
    model_clean = Backbone_Event(args).to(device)
    model_clean.eval()

    # Reset peak video memory statistics
    torch.cuda.reset_peak_memory_stats(device)

    print("Warming up GPU...")
    with torch.no_grad():
        for _ in range(10):
            _ = model_clean(dummy_lidar, dummy_event, 12, test_mode=True)

    print("Measuring Latency (12 iters, 50 runs)...")
    torch.cuda.synchronize() 
    
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(50)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(50)]
    
    with torch.no_grad():
        for i in range(50):
            start_events[i].record()
            _ = model_clean(dummy_lidar, dummy_event, 12, test_mode=True)
            end_events[i].record()
            
    torch.cuda.synchronize() 
    
    # Calculate the mean delay
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_latency_ms = sum(times) / 50.0
    
    # Obtain peak video memory capacity
    peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    peak_mem_mb = peak_mem_bytes / (1024 * 1024)

    print(f"Average Runtime (12 iters): {avg_latency_ms:.2f} ms")
    print(f"Peak VRAM Memory Allocated: {peak_mem_mb:.2f} MB")

if __name__ == '__main__':
    benchmark()