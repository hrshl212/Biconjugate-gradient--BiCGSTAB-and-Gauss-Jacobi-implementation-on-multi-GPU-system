# Biconjugate-gradient-and-Gauss-Jacobi-implementation-on-multi-GPU-system
This repository contains implementation for Biconjugate gradient and Jacobi methods for implicit solution of PDE (heat equation).

This project solves the **Laplace equation**:

∇²u = 0

yaml
Copy
Edit

in a 2D domain using the **Jacobi iterative method** with:

- **Neumann boundary conditions** (zero normal derivative),
- An **initial Gaussian heat spot** as the starting condition,
- Iterative updates until **convergence** (residual < `1e-6`) or a maximum number of iterations.

---

