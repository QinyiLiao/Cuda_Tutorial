# Summary of CUDA Syntax and Concepts

## Overview of CUDA Architecture and Memory Hierarchy

**CUDA (Compute Unified Device Architecture)** is a parallel computing platform and programming model developed by NVIDIA, enabling developers to leverage the GPU's powerful computational capabilities for general-purpose computing.

### Basic Principles

CUDA organizes threads in a hierarchical structure on the GPU as shown in the figure:

![Figure 1](CUDA-memory-model.gif "Memory architecture")

* **Thread**: The smallest execution unit, executes a single instruction stream.
* **Warp**: A group of 32 threads; the basic scheduling unit, executing synchronously in SIMT mode.
* **Block (Thread Block)**: A group of threads. Threads in the same block can synchronize and communicate via shared memory.
* **Grid**: A group of thread blocks, forming a full CUDA kernel execution configuration.

### CUDA Memory Hierarchy

CUDA provides a multi-level memory architecture, each type having different scope, speed, and lifetime:

1. **Registers**

   * Private to each thread and the fastest to access.
   * Limited in number; excessive usage leads to spilling into local memory.

2. **Local Memory**

   * Stores spilled variables and large arrays/structures.
   * Located in global memory; slower despite the name "local".

3. **Shared Memory**

   * Shared among threads in the same block.
   * Faster than global memory but slower than registers.
   * Used for inter-thread communication and data reuse.
   * Declared with `__shared__`.

4. **Global Memory**

   * Accessible by all threads and the host (CPU).
   * Largest but slowest memory.
   * Allocated with `cudaMalloc`, released with `cudaFree`.

5. **Constant Memory**

   * Read-only by all threads.
   * Cached, making it efficient for unchanging data.
   * Declared with `__constant__`.

6. **Unified Memory**

   * Automatically managed memory shared between CPU and GPU.
   * Declared with `__managed__`.

### Memory Access Performance Considerations

* **Coalesced Access**: Threads in a warp accessing consecutive memory addresses can have their accesses combined into fewer transactions, improving bandwidth.
* **Bank Conflicts**: Shared memory is divided into banks; simultaneous accesses to the same bank by different threads cause serialization.
* **Latency Hiding**: The GPU hides memory latency via rapid context switching between warps, requiring high occupancy.

---

## 1. CUDA Function Qualifiers

### `__global__`

* Indicates a kernel function executed on the GPU and callable from the CPU.
* Must return `void`.
* Invoked with triple angle bracket syntax: `func<<<grid, block>>>(args);`
* Example: `__global__ void predic(tpvec *dcon, tpheun *dheun)`

### `__device__`

* Declares a function callable only from GPU code and executed on the GPU.
* Visible across the GPU.
* Example: `__device__ double atomicAdd(double* address, double val)`

### `__constant__`

* Declares a constant memory variable on the GPU.
* Readable by all threads.
* Example: `__constant__ tpbox dbox;`

### `__shared__`

* Declares shared memory visible to all threads in the same block.
* Example: `__shared__ double block_forpre[blocks];`

### `__device__ __managed__`

* Declares a unified memory variable accessible from both CPU and GPU.
* Example: `__device__ __managed__ tpsys dsyst;`

---

## 2. CUDA Memory Management

### Allocation and Deallocation

```cpp
cudaMalloc((void**)&dcon, box.natom * sizeof(tpvec));
cudaFree(dcon);
```

### Memory Copy

```cpp
// Host to Device
cudaMemcpy(dcon, con, box.natom * sizeof(tpvec), cudaMemcpyHostToDevice);

// Copy to constant memory
cudaMemcpyToSymbol(dbox, &box, sizeof(tpbox));
```

### Synchronization

```cpp
cudaDeviceSynchronize(); // Waits for all GPU operations to finish
```

---

## 3. Launching CUDA Kernels

```cpp
dim3 dimblock = blocks; // Set block size
dim3 dimgrid = grids;   // Set grid size
update <<<dimgrid, dimblock>>> (dcon, dnblist); // Launch kernel
```

---

## 4. Thread Index Computation

```cpp
// Compute global thread index
int ni = blockIdx.x * blockDim.x + threadIdx.x;

// Thread index within block
int tid = threadIdx.x;

// Boundary check
if (ni >= dbox.natom) return;
```

---

## 5. Parallel Reduction Pattern

```cpp
// Store thread result in shared memory
block_forpre[tid] = forpre;
__syncthreads(); // Synchronize all threads in block

// Perform reduction within block
int jj = blocks / 2;
while (jj != 0) {
    if (tid < jj) {
        block_forpre[tid] += block_forpre[tid + jj];
    }
    __syncthreads();
    jj /= 2;
}

// First thread updates global result with atomic add
if (tid == 0){
    atomicAdd(&dsyst.forpre, block_forpre[0]);
}
```

---

## 6. Atomic Operations

### `atomicAdd` — Atomic Addition

```cpp
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
```

* Atomic operations ensure correctness when multiple threads access the same memory location concurrently.

---

## 7. `dim3` Type

`dim3` is a CUDA-specific type with three unsigned integers `(x, y, z)` used to specify 1D/2D/3D grid or block dimensions.

```cpp
dim3 dimBlock(16, 16, 1); // 2D thread block with 256 threads
dim3 dimGrid(32, 32, 1);  // 2D grid with 1024 thread blocks
```

When only one dimension is provided, the others default to 1:

```cpp
dim3 dimblock = blocks; // Equivalent to dim3(blocks, 1, 1)
```

---

## 8. Choosing Block and Grid Sizes

### Recommended Block Size (`blockDim`)

* Should be a multiple of 32 (warp size).
* Typically between 128 and 256 threads.
* Larger sizes may reduce occupancy due to resource limits.

### Recommended Grid Size (`gridDim`)

* Usually: `ceil(number_of_elements / block_size)`
* Ensure enough blocks to saturate GPU multiprocessors.

### Best Practice

```cpp
dim3 dimblock = 128;  // or 256
dim3 dimgrid = (box.natom + dimblock - 1) / dimblock;  // dynamic grid sizing
```

### Why Oversized Blocks Hurt Performance

1. **Resource Constraints and Lower Occupancy**

   * Each SM has limited registers and shared memory.
   * Larger blocks may reduce the number of blocks per SM.

2. **Higher Synchronization Overhead**

   * More threads per block → heavier `__syncthreads()` cost.

3. **Warp Scheduling Inefficiency**

   * More warps per block → increased scheduling pressure.

4. **Shared Memory Bank Conflicts**

   * Larger blocks increase chance of access conflicts.

5. **Register Spills**

   * Exceeding register availability spills variables into local memory, reducing performance.

