# Scenario 3: Tensor Batching

This folder demonstrates how to receive a tensor (as a list of floats) via POST, batch multiple tensors together into a higher-dimensional tensor, and process them on TPU.

## Files

- `serve.py`: Ray Serve script handling tensor inputs and batching.
- `simple_client.py`: A client to test the tensor serving endpoint with a single request.
- `simulate-client-requests.py`: Client script to simulate concurrent tensor requests with random delays.

## Steps to Run

### 1. Port-Forward the Dashboard

In a separate terminal, port-forward the Ray dashboard service:
```bash
kubectl port-forward service/ray-tpu-singlehost-cluster-head-svc 8265:8265
```

### 2. Submit the Serve Job

Set the `RAY_ADDRESS` and submit the job. Run this command from the `ray-serve-tpu/03_tensor_batching` directory:

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

To test with a single request:
```bash
python3 simple_client.py
```

To simulate a realistic load with staggered arrival times:
```bash
python3 simulate-client-requests.py 10
```
