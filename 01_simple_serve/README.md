# Scenario 1: Simple JAX Serving

This folder contains a simple example of serving a JAX workload on a TPU v7x (4 chips / 8 cores) using Ray Serve.

## Files

- `serve.py`: The Ray Serve application script.
- `simple_client.py`: A simple python client to test the service.

## Steps to Run

### 1. Port-Forward the Dashboard

In a separate terminal, port-forward the Ray dashboard service:
```bash
kubectl port-forward service/ray-tpu-singlehost-cluster-head-svc 8265:8265
```

### 2. Submit the Serve Job

Set the `RAY_ADDRESS` and submit the job. Run this command from the `ray-serve-tpu/01_simple_serve` directory:

```bash
export RAY_ADDRESS="http://127.0.0.1:8265"
ray job submit --working-dir . -- serve run serve:deployment
```

### 3. Port-Forward for Inference

Ray Serve starts an HTTP proxy on port `8000` on the head node. Port-forward the head pod directly (replace with your actual pod name):

```bash
kubectl port-forward pod/<head-pod-name> 8000:8000
```

### 4. Test the Service

```bash
python3 simple_client.py 5.0
```
