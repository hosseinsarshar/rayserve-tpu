"""Ray Serve script for a JAX workload with tensor inputs and dynamic batching.

This script demonstrates how to receive a tensor (as a list of floats) via POST,
batch multiple tensors together into a higher-dimensional tensor, and process
them on TPU.
"""

import logging
from typing import List
from fastapi import FastAPI
import jax
import jax.numpy as jnp
from pydantic import BaseModel
from ray import serve
import time

app = FastAPI()
logger = logging.getLogger("ray.serve")

MAX_BATCH_SIZE = 8
TENSOR_SIZE = 10  # Fixed size for the input tensor

class TensorInput(BaseModel):
    data: List[float]

@serve.deployment(
    ray_actor_options={
        "resources": {"TPU": 4},
    },
)
class TensorJAXServe:
    def __init__(self):
        logger.info("Initializing Tensor JAX Serve on TPU.")
        logger.info(f"Available devices: {jax.devices()}")
        self.num_devices = jax.device_count()
        
        assert MAX_BATCH_SIZE % self.num_devices == 0, \
            f"MAX_BATCH_SIZE ({MAX_BATCH_SIZE}) must be a multiple of num_devices ({self.num_devices})"
            
        # Simple operation: add 1.0 to all elements of the tensor
        @jax.pmap
        def process_tensor(x):
            return x + 1.0
            
        self.process_tensor = process_tensor

    @serve.batch(batch_wait_timeout_s=0.5, max_batch_size=MAX_BATCH_SIZE)
    async def handle_batch(self, requests: List[dict]):
        current_time = time.time()
        batch_size = len(requests)
        
        # Calculate wait times for each request
        wait_times = [current_time - r["arrival_time"] for r in requests]
        avg_wait = sum(wait_times) / batch_size if batch_size > 0 else 0
        
        logger.info(f"--- Dynamic Batching Stats ---")
        logger.info(f"Tensors arrived in this batch: {batch_size}")
        logger.info(f"Wait times (seconds) -> Min: {min(wait_times):.4f}, Max: {max(wait_times):.4f}, Avg: {avg_wait:.4f}")
        
        orig_len = batch_size
        num_to_pad = MAX_BATCH_SIZE - orig_len
        logger.info(f"Padding size added: {num_to_pad} (to reach fixed size {MAX_BATCH_SIZE})")
        logger.info(f"-------------------------------")
        
        # Extract tensors from requests
        tensors = [r["data"] for r in requests]
        
        # Pad with zero-tensors to maintain fixed shape
        zero_tensor = [0.0] * TENSOR_SIZE
        padded_tensors = tensors + [zero_tensor] * num_to_pad
        
        # Convert to JAX array (Shape: MAX_BATCH_SIZE, TENSOR_SIZE)
        data = jnp.array(padded_tensors)
        
        # Reshape for pmap (Shape: num_devices, items_per_device, TENSOR_SIZE)
        items_per_device = MAX_BATCH_SIZE // self.num_devices
        data = data.reshape((self.num_devices, items_per_device, TENSOR_SIZE))
        
        # Run on TPU
        result = self.process_tensor(data)
        result.block_until_ready()
        
        # Flatten back to (MAX_BATCH_SIZE, TENSOR_SIZE)
        flat_result = result.reshape((MAX_BATCH_SIZE, TENSOR_SIZE))
        
        # Remove padding before returning
        final_result = flat_result[:orig_len].tolist()
        
        return final_result

    async def predict(self, data: List[float]):
        arrival_time = time.time()
        return await self.handle_batch({"data": data, "arrival_time": arrival_time})

@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, model_handle):
        self.model_handle = model_handle

    @app.post("/process_tensor")
    async def process_tensor(self, input_data: TensorInput):
        if len(input_data.data) != TENSOR_SIZE:
            return {"error": f"Input tensor must have size {TENSOR_SIZE}, got {len(input_data.data)}"}
            
        result = await self.model_handle.predict.remote(input_data.data)
        return {"result": result}

# Bind the deployments
model_bound = TensorJAXServe.bind()
deployment = APIIngress.bind(model_bound)
