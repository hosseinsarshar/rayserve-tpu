"""Ray Serve script for a JAX workload with variable input shapes and dynamic batching.

This script demonstrates how to handle variable sequence lengths by padding to
pre-compiled bucket sizes. It uses JAX's compilation cache to store graphs on
disk (/tmp/jax_cache) to avoid recompilation overhead.
"""

import logging
import os
from typing import List
from fastapi import FastAPI
import jax
import jax.numpy as jnp
from pydantic import BaseModel
from ray import serve
import time
import numpy as np


app = FastAPI()
logger = logging.getLogger("ray.serve")

MAX_BATCH_SIZE = 8
MAX_LEN = 100  # Default maximum sequence length

class VariableTensorInput(BaseModel):
    data: List[float]

def get_bucket_sizes(max_len):
    """Generates bucket sizes: powers of 2 from 2^3 up to max_len, plus max_len."""
    buckets = []
    size = 8  # 2^3
    while size < max_len:
        buckets.append(size)
        size *= 2
    if max_len not in buckets:
        buckets.append(max_len)
    return sorted(buckets)

@serve.deployment(
    ray_actor_options={
        "resources": {"TPU": 4},
    },
)
class VariableShapeServe:
    def __init__(self, max_len: int = MAX_LEN):
        # Configure JAX compilation cache to save graphs on /tmp/ path
        CACHE_DIR = "/tmp/jax_cache"
        os.makedirs(CACHE_DIR, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", CACHE_DIR)
        jax.config.update("jax_log_compiles", True)

        # Ensure all compiles are logged and cached regardless of size/time
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", -1)

        logger.info("Initializing Variable Shape Serve on TPU.")
        logger.info(f"Available devices: {jax.devices()}")
        self.num_devices = jax.device_count()
        self.max_len = max_len
        
        assert MAX_BATCH_SIZE % self.num_devices == 0
        
        # Complex operation to make compilation intentionally slow.
        # We use a Python loop instead of jax.lax.fori_loop here because JAX 
        # unrolls Python loops during tracing. This creates a massive graph for 
        # 200 iterations, making XLA compilation take a long time. This is useful
        # to showcase the benefits of pre-compilation and caching.
        @jax.pmap
        def process_tensor(x):
            y = x
            for _ in range(200):
                y = (jnp.sin(y) + jnp.cos(y)) * 0.99
            return y
            
        self.process_tensor = process_tensor
        
        # Pre-compile for all bucket sizes
        self.bucket_sizes = get_bucket_sizes(self.max_len)
        logger.info(f"Bucket sizes for pre-compilation: {self.bucket_sizes}")
        
        items_per_device = MAX_BATCH_SIZE // self.num_devices
        
        for size in self.bucket_sizes:
            logger.info(f"Pre-compiling for shape (8, 1, {size})...")
            start_time = time.perf_counter()
            # Create dummy data for compilation
            dummy_data = jnp.zeros((self.num_devices, items_per_device, size))
            # Trigger compilation
            res = self.process_tensor(dummy_data)
            res.block_until_ready()
            duration = time.perf_counter() - start_time
            logger.info(f"Finished pre-compiling for size {size} in {duration:.4f} seconds")

    @serve.batch(batch_wait_timeout_s=0.5, max_batch_size=MAX_BATCH_SIZE)
    async def handle_batch(self, requests: List[dict]):
        current_time = time.time()
        batch_size = len(requests)
        
        wait_times = [current_time - r["arrival_time"] for r in requests]
        avg_wait = sum(wait_times) / batch_size if batch_size > 0 else 0
        
        # Find max length in this batch
        tensors = [r["data"] for r in requests]
        max_len_in_batch = max(len(t) for t in tensors)
        
        # Find the smallest bucket size that can fit the max length
        bucket_size = self.max_len
        for size in self.bucket_sizes:
            if size >= max_len_in_batch:
                bucket_size = size
                break
                
        logger.info(f"--- Dynamic Batching & Shape Stats ---")
        logger.info(f"Tensors arrived: {batch_size}")
        logger.info(f"Max length in batch: {max_len_in_batch} -> Selected bucket: {bucket_size}")
        logger.info(f"Wait times (seconds) -> Min: {min(wait_times):.4f}, Max: {max(wait_times):.4f}, Avg: {avg_wait:.4f}")
        
        orig_len = batch_size
        num_to_pad = MAX_BATCH_SIZE - orig_len
        logger.info(f"Padding batch dimension by: {num_to_pad}")
        logger.info(f"---------------------------------------")
        
        # Pad tensors to the selected bucket size
        padded_tensors = []
        for t in tensors:
            padded_tensors.append(t + [0.0] * (bucket_size - len(t)))
            
        # Pad the batch to MAX_BATCH_SIZE
        zero_tensor = [0.0] * bucket_size
        padded_tensors += [zero_tensor] * num_to_pad
        
        # Convert to JAX array (Shape: MAX_BATCH_SIZE, bucket_size)
        data = jnp.array(padded_tensors)
        
        # Reshape for pmap (Shape: num_devices, items_per_device, bucket_size)
        items_per_device = MAX_BATCH_SIZE // self.num_devices
        data = data.reshape((self.num_devices, items_per_device, bucket_size))
        
        # Log the shape to verify it matches the compiled bucket size
        logger.info(f"Invoking JAX with shape: {data.shape}")
        
        # Run on TPU (JAX will automatically retrieve the cached graph)
        result = self.process_tensor(data)
        result.block_until_ready()
        
        # Flatten back to (MAX_BATCH_SIZE, bucket_size)
        flat_result = result.reshape((MAX_BATCH_SIZE, bucket_size))
        
        # Convert to NumPy array on CPU to avoid JAX compilation during slicing
        flat_result_np = np.array(flat_result)
        
        # Remove padding before returning
        final_result = []
        for i in range(orig_len):
            orig_tensor_len = len(requests[i]["data"])
            final_result.append(flat_result_np[i, :orig_tensor_len].tolist())
            
        return final_result

    async def predict(self, data: List[float]):
        arrival_time = time.time()
        return await self.handle_batch({"data": data, "arrival_time": arrival_time})

@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, model_handle, max_len: int = MAX_LEN):
        self.model_handle = model_handle
        self.max_len = max_len

    @app.post("/process_variable_tensor")
    async def process_variable_tensor(self, input_data: VariableTensorInput):
        if len(input_data.data) > self.max_len:
            return {"error": f"Input tensor exceeds max length {self.max_len}, got {len(input_data.data)}"}
            
        result = await self.model_handle.predict.remote(input_data.data)
        return {"result": result}

# App builder to allow setting max_len via CLI
def app_builder(args: dict):
    max_len = int(args.get("max_len", MAX_LEN))
    logger.info(f"Building app with max_len={max_len}")
    model_bound = VariableShapeServe.bind(max_len=max_len)
    return APIIngress.bind(model_bound, max_len=max_len)
