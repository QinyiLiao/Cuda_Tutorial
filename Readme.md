# CUDA Programming Tutorial

## Overview

This repository provides an introduction to CUDA (Compute Unified Device Architecture) programming for GPU-accelerated scientific computing. The project includes a comprehensive guide to CUDA syntax and concepts, along with a practical example of molecular dynamics simulation using CUDA.

## Contents

- `Cuda_Introduction.md`: A concise tutorial covering CUDA architecture, memory hierarchy, and programming patterns
- `Cuda.cpp`: A working example of a molecular dynamics simulation for active matter using CUDA parallelization

## CUDA Fundamentals

CUDA is NVIDIA's parallel computing platform and programming model that enables developers to use NVIDIA GPUs for general-purpose computing. Key concepts covered in this project:

- **Thread hierarchy**: Threads, warps, blocks, and grids
- **Memory hierarchy**: Registers, shared memory, global memory, constant memory, unified memory
- **Function qualifiers**: `__global__`, `__device__`, `__host__`, `__shared__`, `__constant__`
- **Kernel launches**: How to configure and launch GPU kernels
- **Parallel reduction**: Efficient techniques for summing values across threads
- **Atomic operations**: Safe concurrent memory access
- **Performance optimization**: Coalesced memory access, bank conflict avoidance, occupancy considerations

## Example Application: Active Brownian Particle Simulation

The included `Cuda.cpp` file demonstrates a molecular dynamics simulation of Active Brownian Particles (ABPs) at zero temperature and in the infinite persistence limit. In this regime, particles maintain their self-propulsion direction indefinitely and thermal fluctuations are absent. The simulation:

- Models a 2D system of bidisperse (two sizes) particles with directed self-propulsion
- Implements efficient neighbor list algorithm for collision detection
- Uses the Heun integrator for numerical time evolution (second-order accuracy)
- Parallelizes computation using CUDA for significant performance improvements
- Calculates system properties like pressure and potential energy

### Key Components in the Example

1. **Data Structures**:
   - `tpbox`: Simulation box parameters (dimensions, physical properties)
   - `tpvec`: Particle position and force data
   - `tpheun`: Variables for Heun integration
   - `tplist`: Neighbor list for efficient force calculations

2. **CUDA Kernels**:
   - `cal_force`: Calculate repulsive forces between particles
   - `make_list`: Build neighbor list for collision detection
   - `predic` & `correc`: Implementation of the Heun integrator
   - `check_list`: Determine when to update neighbor lists

3. **Optimization Techniques**:
   - Shared memory for parallel reduction
   - Efficient thread block configuration
   - Coalesced memory access patterns
   - Atomic operations for thread-safe updates

## Getting Started

### Prerequisites

- NVIDIA CUDA Toolkit (version 10.0 or higher recommended)
- NVIDIA GPU with compute capability 3.0 or higher
- C++ compiler compatible with CUDA

### Compilation

To compile the example code:

```bash
nvcc -O3 Cuda.cpp -o active_particles
```

### Execution

```bash
./active_particles
```

## Parameters

The simulation parameters in `Cuda.cpp` can be modified:

- `box.natom`: Number of particles (default: 4096)
- `box.phi`: Volume fraction (default: 0.84)
- `box.ratio`: Diameter ratio of large to small particles (default: 1.4)
- `box.fd`: Self-propulsion force magnitude (default: 1e-3)
- `box.dt`: Time step size (default: 0.1)
- `contr.nperiod`: Number of simulation periods (default: 10000)

## Performance Considerations

For optimal performance:

- Choose block size as a multiple of warp size (32)
- Balance block size and grid size based on your GPU architecture
- Minimize global memory transfers and maximize shared memory usage
- Avoid thread divergence within warps
- Use appropriate atomic operations for thread synchronization

## Further Resources

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)

## License

This project is provided as an educational resource and is licensed under the MIT License.
