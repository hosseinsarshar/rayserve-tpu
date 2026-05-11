"""Ray Serve script for a JAX workload with dynamic batching on TPU v7x.

This script demonstrates how to use Ray Serve's @serve.batch decorator to
dynamically batch requests and process them efficiently on TPU. It uses
padding to ensure fixed shapes for JAX to avoid recompilation.
"""

import logging
from typing import List
from fastapi import FastAPI
import jax
import jax.numpy as jnp
from ray import serve
import time

app = FastAPI()
logger = logging.getLogger("ray.serve")

# Define the max batch size. For JAX on TPU, this should typically be a
# multiple of the number of devices to ensure even sharding.
MAX_BATCH_SIZE = 8 

@serve.deployment(
    ray_actor_options={
        "resources": {"TPU": 4},
    },
)
class BatchedJAXServe:
    def __init__(self):
        logger.info("Initializing Batched JAX Serve on TPU.")
        logger.info(f"Available devices: {jax.devices()}")
        self.num_devices = jax.device_count()
        
        # Ensure MAX_BATCH_SIZE is a multiple of num_devices for simple sharding
        assert MAX_BATCH_SIZE % self.num_devices == 0, \
            f"MAX_BATCH_SIZE ({MAX_BATCH_SIZE}) must be a multiple of num_devices ({self.num_devices})"
            
        # Define a simple function to run on TPU
        @jax.pmap
        def add_one(x):
            return x + 1.0
            
        self.add_one = add_one

    # Decorator to enable dynamic batching.
    # batch_wait_timeout_s: How long to wait for a full batch before processing.
    # max_batch_size: Maximum number of requests to batch together.
    @serve.batch(batch_wait_timeout_s=0.5, max_batch_size=MAX_BATCH_SIZE)
    async def handle_batch(self, requests: List[dict]):
        current_time = time.time()
        batch_size = len(requests)
        
        # Calculate wait times for each request
        wait_times = [current_time - r["arrival_time"] for r in requests]
        avg_wait = sum(wait_times) / batch_size if batch_size > 0 else 0
        
        logger.info(f"--- Dynamic Batching Stats ---")
        logger.info(f"Requests arrived in this batch: {batch_size}")
        logger.info(f"Wait times (seconds) -> Min: {min(wait_times):.4f}, Max: {max(wait_times):.4f}, Avg: {avg_wait:.4f}")
        
        orig_len = batch_size
        num_to_pad = MAX_BATCH_SIZE - orig_len
        logger.info(f"Padding size added: {num_to_pad} (to reach fixed size {MAX_BATCH_SIZE})")
        logger.info(f"-------------------------------")
        
        # Extract values for processing
        values = [r["value"] for r in requests]
        padded_values = values + [0.0] * num_to_pad
        
        # 2. Convert to JAX array
        data = jnp.array(padded_values)
        
        # 3. Reshape for pmap (leading dimension must be num_devices)
        items_per_device = MAX_BATCH_SIZE // self.num_devices
        data = data.reshape((self.num_devices, items_per_device))
        
        # 4. Run on TPU
        result = self.add_one(data)
        result.block_until_ready()
        
        # 5. Flatten and remove padding before returning
        flat_result = result.reshape((MAX_BATCH_SIZE,))
        final_result = flat_result[:orig_len].tolist()
        
        return final_result

    async def predict(self, value: float):
        # Record arrival time to showcase wait time in batch
        arrival_time = time.time()
        # Pass a dict to the batch handler
        return await self.handle_batch({"value": value, "arrival_time": arrival_time})

@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, model_handle):
        self.model_handle = model_handle

    @app.get("/add_one")
    async def add_one(self, value: float = 1.0):
        # Ray Serve will batch these individual calls
        result = await self.model_handle.predict.remote(value)
        return {"result": result}

# Bind the deployments
model_bound = BatchedJAXServe.bind()
deployment = APIIngress.bind(model_bound)
