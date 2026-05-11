# Dynamic Batching with Dynamic Shapes in JAX

Serving models with variable input shapes (like text generation or audio processing) presents a unique challenge in JAX. XLA (the compiler used by JAX) requires fixed shapes to generate optimized machine code for TPUs. If the input shape changes between requests, JAX will trigger a recompilation, which can take seconds or even minutes, causing massive latency spikes.

To handle dynamic batching with dynamic shapes efficiently on TPUs, we use a combination of **Padding**, **Bucketing**, and **Batch Dimension Padding**.

---

## The Concept

To avoid recompilation, we must ensure that the shape passed to the JAX compiled function remains constant across as many requests as possible. We achieve this by mapping variable-sized inputs into a few pre-defined "buckets" and padding the rest.

### Step-by-Step Example

Suppose we have a `MAX_BATCH_SIZE` of 8 and a set of pre-compiled bucket sizes: `[8, 16, 32, 64, 128, 256, 512, 1024]`.

Imagine a scenario where Ray Serve receives a batch of **2 requests** with the following shapes:
*   **Request 1:** Tensor of size `1x8`
*   **Request 2:** Tensor of size `1x120`

Here is how the server processes them:

#### 1. Find the Maximum Length in the Batch
The server looks at all requests in the current batch and finds the maximum sequence length.
*   Max length in this batch = **120**

#### 2. Select the Bucket
The server finds the smallest bucket size that can fit the maximum length.
*   The smallest bucket $\ge$ 120 is **128**.

#### 3. Pad Inputs to the Bucket Size
All inputs in the batch are padded with zeros to match the selected bucket size.
*   Request 1 (`1x8`) is padded to `1x128`.
*   Request 2 (`1x120`) is padded to `1x128`.
*   Now, both inputs have a uniform shape of `1x128`.

#### 4. Pad the Batch Dimension
To handle variable numbers of requests per batch (dynamic batching), the server also pads the batch dimension to the `MAX_BATCH_SIZE`.
*   We have 2 real requests of shape `1x128`.
*   We add 6 "dummy" requests of shape `1x128` filled with zeros.
*   The final shape passed to JAX is **`8x128`** (or `8x1x128`).

#### 5. Execution and Post-processing
JAX receives the fixed shape `8x128` and retrieves the pre-compiled graph from the cache (or compiles it once if not cached). After execution, the server removes the padding and returns the results with their original lengths to the client.

---

## Summary of the Pattern

1.  **Pad to uniform shape within a batch** to allow batch processing.
2.  **Bucket the shapes** to reduce the number of unique shapes JAX sees, minimizing compilations.
3.  **Pad the batch dimension** to handle varying load.
4.  **Pre-compile** for all bucket sizes during initialization to avoid cold-start latency.

This pattern is implemented in the `serve.py` script in this folder.
