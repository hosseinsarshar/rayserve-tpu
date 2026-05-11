"""Simple Ray Serve script for a JAX workload on TPU v7x (4 chips).

This script creates a Ray Serve deployment that runs on a TPU node with 4 chips
(e.g., v7-8 which has 4 chips / 8 cores). It uses a simple pmapped JAX operation
to demonstrate multi-device usage.
"""

import logging
from fastapi import FastAPI
import jax
import jax.numpy as jnp
from ray import serve

app = FastAPI()
logger = logging.getLogger("ray.serve")

# Define the JAX serving deployment
@serve.deployment(
    ray_actor_options={
        # Request 4 TPU chips. For a v7-8 (4 chips), this ensures the actor
        # is placed on a node with sufficient TPU resources.
        "resources": {"TPU": 4},
    },
)
class SimpleJAXServe:
    def __init__(self):
        logger.info("Initializing Simple JAX Serve on TPU.")
        logger.info(f"Available devices: {jax.devices()}")
        self.num_devices = jax.device_count()
        
        # Define a simple function to run on TPU across all devices
        @jax.pmap
        def add_one(x):
            return x + 1.0
            
        self.add_one = add_one

    async def serve_request(self, value: float):
        logger.info(f"Processing request with value: {value}")
        
        # Create data for each device (shard size = num_devices)
        data = jnp.ones((self.num_devices,)) * value
        
        # Run on TPU
        result = self.add_one(data)
        
        # Ensure execution completes before returning
        result.block_until_ready()
        
        return {
            "num_devices": self.num_devices,
            "input_value": value,
            "result": result.tolist(),
        }

# Define the Ingress deployment to handle HTTP requests
@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, model_handle):
        self.model_handle = model_handle

    @app.get("/add_one")
    async def add_one(self, value: float = 1.0):
        result = await self.model_handle.serve_request.remote(value)
        return result

# Bind the deployments together
model_bound = SimpleJAXServe.bind()
deployment = APIIngress.bind(model_bound)
