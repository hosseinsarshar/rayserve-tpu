# Ray Serve on TPU v7x POC

This repository contains a collection of examples demonstrating how to serve JAX workloads on TPU v7x (4 chips / 8 cores) using Ray Serve.

The examples progress in complexity from a simple server to handling dynamic batching, tensor inputs, and variable shapes with compilation caching.

## Shared Cluster Configuration

All examples share the same Ray cluster configuration:

- `ray-gke.yaml`: Kubernetes manifest to create the Ray cluster with TPU v7x workers.

Ensure you apply this manifest first before running any of the scenarios:
```bash
kubectl apply -f ray-gke.yaml
```

## Scenarios

Each scenario is contained in its own folder with its own `README.md` file explaining the details:

1.  **[01_simple_serve](./01_simple_serve/)**: A basic server processing a single number.
2.  **[02_dynamic_batching](./02_dynamic_batching/)**: Demonstrates Ray Serve's dynamic batching with fixed shapes.
3.  **[03_tensor_batching](./03_tensor_batching/)**: Handles receiving and batching 1D tensors (lists of floats).
4.  **[04_variable_shapes](./04_variable_shapes/)**: Advanced example handling variable sequence lengths with bucketing and JAX compilation caching.

Please refer to the `README.md` inside each folder for specific instructions on how to run and test that scenario.

## Cleanup

To stop all serving applications (requires `pip install "ray[serve]"` in local environment):
```bash
export RAY_ADDRESS="http://127.0.0.1:8265"
serve shutdown
```

To delete the entire Ray cluster and release all GKE resources:
```bash
kubectl delete -f ray-gke.yaml
```
