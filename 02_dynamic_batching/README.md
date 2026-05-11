# Scenario 2: Dynamic Batching

This folder demonstrates how to use Ray Serve's `@serve.batch` decorator to dynamically batch requests and process them efficiently on TPU. It uses padding to ensure fixed shapes for JAX to avoid recompilation.

## Files

- `serve.py`: Ray Serve script with dynamic batching enabled.
- `simulate-client-requests.py`: Client script to simulate concurrent requests with random delays.

## Steps to Run

### 1. Port-Forward the Dashboard

In a separate terminal, port-forward the Ray dashboard service:
```bash
kubectl port-forward service/ray-tpu-singlehost-cluster-head-svc 8265:8265
```

### 2. Submit the Serve Job

Set the `RAY_ADDRESS` and submit the job. Run this command from the `ray-serve-tpu/02_dynamic_batching` directory:

```bash
export RAY_ADDRESS="http://127.0.0.1:8265"
ray job submit --working-dir . -- serve run serve:deployment
```

### 3. Port-Forward for Inference

Ray Serve starts an HTTP proxy on port `8000` on the head node. Port-forward the head pod directly (replace with your actual pod name):

```bash
kubectl port-forward pod/<head-pod-name> 8000:8000
```

### 4. Run the Simulation

```bash
python3 simulate-client-requests.py 10
```

Check the Ray Serve logs to see the dynamic batching statistics!
