// multi_gpu_bicg_time.cu - Solves transient heat equation using true BiCGSTAB solver (Krylov) for implicit scheme on multi-GPU (MPI + CUDA)

#include <cstdio>
#include <cmath>
#include <mpi.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <chrono>

#define IDX(i, j, nx) ((i) + (j)*(nx))

__global__
void apply_operator(const float* u, float* Au, int nx, int ny, float dx2, float dy2, float alpha_dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx-1 && j < ny-1) {
        int id = IDX(i, j, nx);
        float uij = u[id];
        float lap = (u[IDX(i-1,j,nx)] + u[IDX(i+1,j,nx)] - 2*uij) / dx2
                  + (u[IDX(i,j-1,nx)] + u[IDX(i,j+1,nx)] - 2*uij) / dy2;
        Au[id] = uij - alpha_dt * lap;
    }
}

void exchange_halo(float* d_field, int nx, int ny_local, int rank, int size) {
    MPI_Request reqs[4];
    int tag0 = 0, tag1 = 1;

    float* send_top = d_field + IDX(0, ny_local - 2, nx);
    float* recv_top = d_field + IDX(0, ny_local - 1, nx);
    float* send_bot = d_field + IDX(0, 1, nx);
    float* recv_bot = d_field + IDX(0, 0, nx);

    int n_req = 0;

    if (rank > 0) {
        MPI_Irecv(recv_bot, nx, MPI_FLOAT, rank - 1, tag0, MPI_COMM_WORLD, &reqs[n_req++]);
        MPI_Isend(send_bot, nx, MPI_FLOAT, rank - 1, tag1, MPI_COMM_WORLD, &reqs[n_req++]);
    }

    if (rank < size - 1) {
        MPI_Irecv(recv_top, nx, MPI_FLOAT, rank + 1, tag1, MPI_COMM_WORLD, &reqs[n_req++]);
        MPI_Isend(send_top, nx, MPI_FLOAT, rank + 1, tag0, MPI_COMM_WORLD, &reqs[n_req++]);
    }

    MPI_Waitall(n_req, reqs, MPI_STATUSES_IGNORE);
}

__global__
void vec_axpby(float* z , const float* x, const float* y, float alpha, float beta, int nx, int ny_local) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny_local;
    if (idx < N) {
        int j = idx / nx;
        if (j > 0 && j < ny_local - 1)  // skip halos
            z[idx] = alpha * x[idx] + beta * y[idx];
    }
}

__global__
void vec_add(float* z, const float* x, const float* y, float alpha, float beta, int nx, int ny_local) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny_local;
    if (idx < N) {
        int j = idx / nx;
        if (j > 0 && j < ny_local - 1)  // skip halos
            z[idx] =  alpha * x[idx] + beta * y[idx];
    }
}

__global__
void dot_product_kernel(const float* x, const float* y, const int* mask, float* partial_sum, int nx, int ny_local, int rank, int size) {
    extern __shared__ float cache[];  // dynamic shared memory
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float temp = 0;
    int total = nx * ny_local;

    if (i < total) temp = mask[i] * x[i] * y[i];

    cache[tid] = temp;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) cache[tid] += cache[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial_sum[blockIdx.x] = cache[0];

}

float dot_product(const float* d_x, const float* d_y, const int* d_mask, float* d_partial, float* h_partial, int nx, int ny_local, int rank, int size) {
    int threads = std::min(256, nx* ny_local);
    int blocks = (nx* ny_local + threads - 1) / threads;
    
    int shared_mem_size = threads * sizeof(float);

    dot_product_kernel<<<blocks, threads, shared_mem_size>>>(d_x, d_y, d_mask, d_partial, nx, ny_local, rank, size);
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < blocks; ++i) sum += h_partial[i];
    float global_sum;
    MPI_Allreduce(&sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    return global_sum;
}

__global__
void enforce_neumann_bc_x(float* u, int nx, int ny_local, int rank, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx) {
        if (rank == 0) u[IDX(i, 0, nx)] = u[IDX(i, 1, nx)];
        if (rank == size - 1) u[IDX(i, ny_local - 1, nx)] = u[IDX(i, ny_local - 2, nx)];
    }
}

__global__
void enforce_neumann_bc_y(float* u, int nx, int ny_local) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < ny_local) {
        u[IDX(0, j, nx)] = u[IDX(1, j, nx)];
        u[IDX(nx - 1, j, nx)] = u[IDX(nx - 2, j, nx)];
    }
}

void enforce_neumann_bc(float* d_u, int nx, int ny_local, int rank, int size) {
    int threads = 256;
    int blocks_x = (nx + threads - 1) / threads;
    int blocks_y = (ny_local + threads - 1) / threads;
    enforce_neumann_bc_x<<<blocks_x, threads>>>(d_u, nx, ny_local, rank, size);
    enforce_neumann_bc_y<<<blocks_y, threads>>>(d_u, nx, ny_local);
}

void initialize_gaussian(float* u, int nx, int ny_local, int global_j_start, int ny_total) {
    float cx = nx / 2;
    float cy = ny_total / 2;
    float sigma = 10.0f;
    for (int j = 0; j < ny_local; ++j) {
        for (int i = 0; i < nx; ++i) {
            float dx = i - cx;
            float dy = (j + global_j_start - 1) - cy;
            u[IDX(i, j, nx)] = expf(-(dx*dx + dy*dy)/(2*sigma*sigma));
        }
    }
}

void save_solution(float* d_u, int nx, int ny_local, int rank, int size, int step, int ny_total) {
    float* h_local = new float[nx * (ny_local - 2)];
    cudaMemcpy(h_local, d_u + nx, nx * (ny_local - 2) * sizeof(float), cudaMemcpyDeviceToHost);
    float* h_global = nullptr;
    if (rank == 0) h_global = new float[nx * ny_total];

    MPI_Gather(h_local, nx * (ny_local - 2), MPI_FLOAT,
               h_global, nx * (ny_total / size), MPI_FLOAT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        char fname[64];
        sprintf(fname, "heat_%04d.npy", step);
        FILE* f = fopen(fname, "wb");
        fwrite(h_global, sizeof(float), nx * ny_total, f);
        fclose(f);
        printf("Saved %s\n", fname);
        delete[] h_global;
    }
    delete[] h_local;
}

void print_mat(float* d_u, int nx, int ny_local, int rank, int size,  int ny_total) {
    float* h_local = new float[nx * (ny_local - 2)];
    cudaMemcpy(h_local, d_u + nx, nx * (ny_local - 2) * sizeof(float), cudaMemcpyDeviceToHost);
    float* h_global = nullptr;
    if (rank == 0) h_global = new float[nx * ny_total];

    MPI_Gather(h_local, nx * (ny_local - 2), MPI_FLOAT,
               h_global, nx * (ny_total / size), MPI_FLOAT,
               0, MPI_COMM_WORLD);
    
    std::cout << "printing mat" << std::endl;
    if(rank == 0){
        for(int i=0; i< nx; i++){
            for(int j=0; j<ny_total; j++){
                std::cout << h_global[IDX(i,j,nx)] << "   ";
            }
            std::cout << std::endl;
        }
    }
    delete[] h_global;
    delete[] h_local;
}

void create_mask(int* mask, int nx, int ny_local, int rank, int size) {
    for (int j = 0; j < ny_local; ++j) {
        for (int i = 0; i < nx; ++i) {
            bool is_boundary = (i == 0 || i == nx - 1 ||
                               (j == 0 && rank == 0) ||
                               (j == ny_local - 1 && rank == size - 1));
            mask[IDX(i, j, nx)] = is_boundary ? 0 : 1;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cudaSetDevice(rank);
    const int nx = 128, ny_total = 128;
    const float alpha_e = 0.1f, dx = 1.0f, dy = 1.0f;
    const float dt = 1.0f, alpha_dt = alpha_e * dt;
    const int steps = 1000, max_iters = 500, save_interval = 40;
    const float tol = 1e-6f;

    int ny_local = ny_total / size + 2;
    int N = nx * ny_local;
    size_t bytes = N * sizeof(float);
    int* h_mask = new int[nx * ny_local];
    create_mask(h_mask, nx, ny_local, rank, size);

    float *d_x, *d_rhs, *d_r, *d_r0, *d_p, *d_v, *d_Ap, *d_s, *d_tt;
    float *d_partial;
    int* d_mask;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_rhs, bytes);
    cudaMalloc(&d_r, bytes);
    cudaMalloc(&d_r0, bytes);
    cudaMalloc(&d_p, bytes);
    cudaMalloc(&d_v, bytes);
    cudaMalloc(&d_Ap, bytes);
    cudaMalloc(&d_s, bytes);
    cudaMalloc(&d_tt, bytes);
    cudaMalloc(&d_mask, nx * ny_local * sizeof(int));
    cudaMalloc(&d_partial, ((N + 255) / 256) * sizeof(float));
    float* h_partial = new float[(N + 255) / 256];
    cudaMemcpy(d_mask, h_mask, nx * ny_local * sizeof(int), cudaMemcpyHostToDevice);
    delete[] h_mask;

    float* h_u_init = new float[N];
    initialize_gaussian(h_u_init, nx, ny_local, rank * (ny_total / size), ny_total);
    cudaMemcpy(d_x, h_u_init, bytes, cudaMemcpyHostToDevice);
    delete[] h_u_init;

    std::cout << std::fixed << std::setprecision(1); // Set output to fixed-point notation with 1 decimal places
    // print_mat(d_x,  nx,  ny_local,  rank,  size,  ny_total);

    int threads_one_dir = 16;
    dim3 threads(threads_one_dir, threads_one_dir);
    dim3 blocks((nx - 2 + threads_one_dir-1) / threads_one_dir, (ny_local - 2 + threads_one_dir-1) / threads_one_dir);
    dim3 threads1D(threads_one_dir*threads_one_dir);
    dim3 blocks1D((N + (threads_one_dir*threads_one_dir-1)) / (threads_one_dir*threads_one_dir));

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t <= steps; ++t) {
        cudaMemcpy(d_rhs, d_x, bytes, cudaMemcpyDeviceToDevice);

        // cudaMemset(d_x, 0, bytes);  // or d_x = d_rhs if warm-starting
        // r = b - A*u
        exchange_halo(d_x, nx, ny_local, rank, size);    
        enforce_neumann_bc(d_x, nx, ny_local, rank, size);
        apply_operator<<<blocks, threads>>>(d_x, d_Ap, nx, ny_local, dx*dx, dy*dy, alpha_dt);

        vec_add<<<blocks1D, threads1D>>>(d_r, d_rhs, d_Ap, 1.0f, -1.0f,  nx, ny_local);

        cudaMemcpy(d_r0, d_r, bytes, cudaMemcpyDeviceToDevice); //d_r0 is the shadow residual
        cudaMemcpy(d_p, d_r, bytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_v, d_r, bytes, cudaMemcpyDeviceToDevice); 

        float rho_old = 1.0f, alpha = 1.0f, omega = 1.0f;
        float beta, rho_new, resid, denom;
        
        if (t % save_interval == 0)
            save_solution(d_x, nx, ny_local, rank, size, t, ny_total);

        int k = 0;
        while (k < max_iters) {
            rho_new = dot_product(d_r0, d_r, d_mask, d_partial, h_partial, nx, ny_local, rank, size);
            beta = (rho_new / rho_old) * (alpha / omega);

            // p = r + beta * (p - omega * v)
            vec_axpby<<<blocks1D, threads1D>>>(d_p, d_p, d_v, 1.0f, -omega, nx, ny_local);
            vec_axpby<<<blocks1D, threads1D>>>(d_p, d_p, d_r, beta, 1.0f,  nx, ny_local);


            // v = A * p
            exchange_halo(d_p, nx, ny_local, rank, size);    
            enforce_neumann_bc(d_p, nx, ny_local, rank, size);
            apply_operator<<<blocks, threads>>>(d_p, d_v, nx, ny_local, dx*dx, dy*dy, alpha_dt);

            denom = dot_product(d_r0, d_v, d_mask, d_partial, h_partial, nx, ny_local, rank, size);
            alpha = rho_new / denom;

            // s = r - alpha * v
            vec_axpby<<<blocks1D, threads1D>>>(d_s, d_r, d_v, 1.0f, -alpha,  nx, ny_local);   

            // t = A * s
            exchange_halo(d_s, nx, ny_local, rank, size);    
            enforce_neumann_bc(d_s, nx, ny_local, rank, size);
            apply_operator<<<blocks, threads>>>(d_s, d_tt, nx, ny_local, dx*dx, dy*dy, alpha_dt); 

            omega = dot_product(d_tt, d_s, d_mask, d_partial, h_partial, nx, ny_local, rank, size) / dot_product(d_tt, d_tt, d_mask, d_partial, h_partial, nx, ny_local, rank, size);

            // x = x + alpha * p + omega * s
            vec_axpby<<<blocks1D, threads1D>>>(d_p, d_p, d_s, alpha, omega, nx, ny_local);
            vec_axpby<<<blocks1D, threads1D>>>(d_x, d_x, d_p, 1.0f, 1.0f, nx, ny_local);
    
            // r = s - omega * t
            vec_axpby<<<blocks1D, threads1D>>>(d_r, d_s, d_tt, 1.0f, -omega, nx, ny_local);
    
            resid = sqrt(dot_product(d_r, d_r, d_mask, d_partial, h_partial, nx, ny_local, rank, size));
            // printf("Iteration %d, Residual = %e\n", k, resid);
            if (resid < tol) break;
    
            rho_old = rho_new;
            k++;
        }

        // if (rank == 0 && t % save_interval == 0)
        //     printf("[t=%d] BiCG converged in %d iterations\n", t, k);
    }

    // Record the end time
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Print the execution time
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;

    cudaFree(d_x); cudaFree(d_rhs); cudaFree(d_r); cudaFree(d_r0);
    cudaFree(d_p); cudaFree(d_v); cudaFree(d_Ap); cudaFree(d_s); cudaFree(d_partial);
    cudaFree(d_mask); cudaFree(d_tt);
    delete[] h_partial;
    MPI_Finalize();
    return 0;
}

