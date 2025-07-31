# Biconjugate gradient, BiCGSTAB and Gauss-Jacobi implementation on multi-GPU system
This repository contains implementation for Biconjugate gradient method and Jacobi method for implicit solution of PDE (heat equation). Both the solvers are for multi-GPU configuration and were run on NVIDIA A100s in rockfish cluster at Johns Hopkins University.

This project solves the **Heat equation**:

∂u/∂t = ∇²u

in a 2D domain using the **BiCG and Jacobi iterative methods** with:

- **Neumann boundary conditions** (zero normal derivative),
- An **initial Gaussian heat spot** as the starting condition,
- Iterative updates until **convergence** (residual < `1e-6`) or a maximum number of iterations.

---
## 🌡️ Initial Condition

We use a **centered Gaussian** as the initial distribution:

u(x, y, t=0) = exp(-((x - x₀)² + (y - y₀)²) / (2σ²))

Where:
- `(x₀, y₀)` is the center of the domain,
- `σ` controls the spread of the heat spot.

---
## 🛠️ How to Run

```bash
nvcc -c -o multigpu_bicg.o multigpu_bicg.cu
mpicxx -o multigpu_bicg multigpu_bicg.o -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 -lcudart -lmpi -lpmix
```
The jobscript file is used to submit the job on JHU's rockfish cluster

---
## Implementation Details

– The multi-GPU solvers are implemented by discretizing the domain among different GPUs
and with one thread taking care of one grid point.

– Each MPI rank manages 1 GPU

– CUDA-aware MPI is utilized to communicate information between the GPUs in the ghost or halo region. This avoides unnecessary copies to host and enables direct communication between the GPUs.

– A mask is implemented to identify internal grid points, boundary points and ghost points (or halo region). This mask
is utilized inside CUDA kernels to avoid if-statements (Warp divergence) thereby aiding in optimized kernel
development.

---
## Results
For a discretization of size 128 × 128 grid points, the Gauss-Jacobi solver requires 0.44 seconds, the BiCG solver takes 0.49 seconds, and the BiCGSTAB solver takes 0.47 seconds to complete 1000 timesteps. In contrast, for a larger discretization of 1024 × 1024 grid points, the Gauss-Jacobi solver takes 6.05 seconds, the BiCG solver takes 0.82 seconds, and the BiCGSTAB solver completes the same number of timesteps in 0.58 seconds. These results indicate that for smaller grids, all three solvers perform comparably, with Gauss-Jacobi showing a slight advantage. However, for larger grids, the BiCGSTAB solver demonstrates a significant performance improvement over the others.

<figure>
  <img src="./temp_evolution_jacobi.gif" alt="Gauss-Jacobi method"/>
  <figcaption>Gauss-Jacobi method</figcaption>
</figure>

<figure>
  <img src="./temp_evolution_jacobi.gif" alt="Biconjugate gradient method"/>
  <figcaption>Biconjugate gradient method</figcaption>
</figure>

---
## Visualization
Run animate_heat.py to generate the gif visualizing the evolution of u
