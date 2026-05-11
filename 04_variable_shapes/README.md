# Scenario 4: Variable Shapes & Caching

This folder demonstrates how to handle variable sequence lengths by padding to pre-compiled bucket sizes. It uses JAX's compilation cache to store graphs on disk (`/tmp/jax_cache`) to avoid recompilation overhead.

## Files

- `serve.py`: Ray Serve script handling variable shapes and caching.
- `simulate-client-requests.py`: Client script to simulate requests with variable shapes.
- `compilation_test.py`: Script to test JAX compilation caching independently.
## Concept: Dynamic Batching with Dynamic Shapes in JAX

Serving models with variable input shapes (like text generation or audio processing) presents a unique challenge in JAX. XLA (the compiler used by JAX) requires fixed shapes to generate optimized machine code for TPUs. If the input shape changes between requests, JAX will trigger a recompilation, which can take seconds or even minutes, causing massive latency spikes.

To handle dynamic batching with dynamic shapes efficiently on TPUs, we use a combination of **Padding**, **Bucketing**, and **Batch Dimension Padding**.

---

### The Concept

To avoid recompilation, we must ensure that the shape passed to the JAX compiled function remains constant across as many requests as possible. We achieve this by mapping variable-sized inputs into a few pre-defined "buckets" and padding the rest.

#### Step-by-Step Example

Suppose we have a `MAX_BATCH_SIZE` of 8 and a set of pre-compiled bucket sizes: `[8, 16, 32, 64, 128, 256, 512, 1024]`.

Imagine a scenario where Ray Serve receives a batch of **2 requests** with the following shapes:
*   **Request 1:** Tensor of size `1x8`
*   **Request 2:** Tensor of size `1x120`

Here is how the server processes them:

##### 1. Find the Maximum Length in the Batch
The server looks at all requests in the current batch and finds the maximum sequence length.
*   Max length in this batch = **120**

##### 2. Select the Bucket
The server finds the smallest bucket size that can fit the maximum length.
*   The smallest bucket $\ge$ 120 is **128**.

##### 3. Pad Inputs to the Bucket Size
All inputs in the batch are padded with zeros to match the selected bucket size.
*   Request 1 (`1x8`) is padded to `1x128`.
*   Request 2 (`1x120`) is padded to `1x128`.
*   Now, both inputs have a uniform shape of `1x128`.

##### 4. Pad the Batch Dimension
To handle variable numbers of requests per batch (dynamic batching), the server also pads the batch dimension to the `MAX_BATCH_SIZE`.
*   We have 2 real requests of shape `1x128`.
*   We add 6 "dummy" requests of shape `1x128` filled with zeros.
*   The final shape passed to JAX is **`8x128`** (or `8x1x128`).

##### 5. Execution and Post-processing
JAX receives the fixed shape `8x128` and retrieves the pre-compiled graph from the cache (or compiles it once if not cached). After execution, the server removes the padding and returns the results with their original lengths to the client.

---

### Summary of the Pattern

1.  **Pad to uniform shape within a batch** to allow batch processing.
2.  **Bucket the shapes** to reduce the number of unique shapes JAX sees, minimizing compilations.
3.  **Pad the batch dimension** to handle varying load.
4.  **Pre-compile** for all bucket sizes during initialization to avoid cold-start latency.

This pattern is implemented in the `serve.py` script in this folder.

## Steps to Run

### 1. Port-Forward the Dashboard

In a separate terminal, port-forward the Ray dashboard service:
```bash
kubectl port-forward service/ray-tpu-singlehost-cluster-head-svc 8265:8265
```

### 2. Submit the Serve Job

Set the `RAY_ADDRESS` and submit the job. Run this command from the `ray-serve-tpu/04_variable_shapes` directory:

```bash
export RAY_ADDRESS="http://127.0.0.1:8265"
ray job submit --working-dir . -- serve run serve:app_builder max_len=200
```

This service uses an `app_builder` to allow setting `max_len` via CLI (default is 100).

### 3. Port-Forward for Inference

Ray Serve starts an HTTP proxy on port `8000` on the head node. Port-forward the head pod directly (replace with your actual pod name):

```bash
kubectl port-forward pod/<head-pod-name> 8000:8000
```

### 4. Run the Simulation

```bash
python3 simulate-client-requests.py 10
```

## What to Expect: Logs

When you run the service and the simulation client, you should see the following behavior in the Ray Serve logs:

### 1. Pre-compilation and Cache Hits on Startup

When the server starts, it will pre-compile graphs for all bucket sizes up to `max_len`. You will see logs indicating whether it is a fresh compilation or a **cache hit** from the persistent cache:

**Example of a Cache Hit (Extremely Fast):**
```
(ServeReplica:default:VariableShapeServe pid=...) Persistent compilation cache hit for 'jit_broadcast_in_dim' with key '...'
(ServeReplica:default:VariableShapeServe pid=...) Finished XLA compilation of jit(broadcast_in_dim) in 0.0011 sec
```
Notice that when there is a cache hit, the "compilation" time is just a few milliseconds because JAX is simply fetching the graph from the `/tmp/jax_cache` directory.

**Example of Pre-compilation completion:**
```
(ServeReplica:default:VariableShapeServe pid=...) INFO ... -- Finished pre-compiling for size 8 in 0.7955 seconds
(ServeReplica:default:VariableShapeServe pid=...) INFO ... -- Finished pre-compiling for size 16 in 0.6136 seconds
...
(ServeReplica:default:VariableShapeServe pid=...) INFO ... -- Finished pre-compiling for size 128 in 0.6089 seconds
...
```
These times include the total time for tracing, potential compilation (if missed), and execution.

### 2. Zero Compilation during Requests
Once requests start arriving, the dynamic batcher groups them and selects the appropriate bucket. Because we pre-compiled the shapes, you will see **zero compilation logs** during inference requests!

```
(ServeReplica:default:VariableShapeServe pid=...) INFO ... -- --- Dynamic Batching & Shape Stats ---
(ServeReplica:default:VariableShapeServe pid=...) INFO ... -- Tensors arrived: 5
(ServeReplica:default:VariableShapeServe pid=...) INFO ... -- Max length in batch: 95 -> Selected bucket: 128
...
(ServeReplica:default:VariableShapeServe pid=...) Persistent compilation cache hit for 'jit_convert_element_type' with key '...'
(ServeReplica:default:VariableShapeServe pid=...) Finished XLA compilation of jit(convert_element_type) in 0.0017 sec
...
(ServeReplica:default:VariableShapeServe pid=...) INFO ... -- Invoking JAX with shape: (8, 1, 128)
```

Notice that the compilation time is just **0.0017 seconds** (due to the cache hit), instead of taking **hundreds of milliseconds or seconds** if it had to compile from scratch. 

Also notice that there are no `Compiling jit(...)` logs for the main graph here, and execution proceeds immediately. This confirms that dynamic batching and shape bucketing are successfully avoiding XLA compilation latency during serving.
```
