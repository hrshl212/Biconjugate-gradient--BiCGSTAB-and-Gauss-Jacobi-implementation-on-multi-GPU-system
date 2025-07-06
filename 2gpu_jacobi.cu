// multi_gpu_jacobi_time.cu - Solves transient heat equation using Jacobi iterations with implicit scheme on multi-GPU (MPI + CUDA) with residual-based convergence per time step

#include <cstdio>
#include <cmath>
#include <mpi.h>
#include <cuda_runtime.h>
#include <chrono>

#define IDX(i, j, nx) ((i) + (j)*(nx))

__global__
void jacobi_step(
    float* u_new, const float* u_old, const float* rhs,
    int nx, int ny, float dx2, float dy2, float alpha_dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i < nx - 1 && j < ny - 1) {
        int id = IDX(i, j, nx);
        float ux = (u_old[IDX(i-1,j,nx)] - 2*u_old[IDX(i,j,nx)] + u_old[IDX(i+1,j,nx)]) / dx2 ;
        float uy = (u_old[IDX(i,j-1,nx)] - 2*u_old[IDX(i,j,nx)] + u_old[IDX(i,j+1,nx)]) / dy2;
        // float denom = dx2 + dy2 + alpha_dt * 2.0f * (dx2 + dy2);
        u_new[id] = rhs[id] + alpha_dt*(ux + uy) ;
    }
}

__global__
void compute_residual(
    const float* u_new, const float* u_old, float* res,
    int nx, int ny
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i < nx - 1 && j < ny - 1) {
        int id = IDX(i, j, nx);
        float diff = u_new[id] - u_old[id];
        res[id] = diff * diff;
    }
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

void exchange_boundaries(float* d_field, int nx, int ny_local, int rank, int size) {
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cudaSetDevice(rank);
    const int nx = 128, ny_total = 128;
    const float alpha = 0.1f, dx = 1.0f, dy = 1.0f;
    const float dt = 1.0f, alpha_dt = alpha * dt;
    const int steps = 1000, max_jacobi_iters = 500, save_interval = 40;
    const float tol = 1e-6f;

    int ny_local = ny_total / size + 2;
    int N = nx * ny_local;
    size_t bytes = N * sizeof(float);

    float *d_u_old, *d_u_new, *d_rhs, *d_res;
    cudaMalloc(&d_u_old, bytes);
    cudaMalloc(&d_u_new, bytes);
    cudaMalloc(&d_rhs, bytes);
    cudaMalloc(&d_res, bytes);

    float* h_u_init = new float[N];
    initialize_gaussian(h_u_init, nx, ny_local, rank * (ny_total / size), ny_total);
    cudaMemcpy(d_u_old, h_u_init, bytes, cudaMemcpyHostToDevice);
    delete[] h_u_init;

    dim3 threads(16, 16);
    dim3 blocks((nx-2 + 15)/16, (ny_local-2 + 15)/16);

    float dx2 = dx*dx, dy2 = dy*dy;

    float* h_res = new float[N];
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t <= steps; ++t) {
        cudaMemcpy(d_rhs, d_u_old, bytes, cudaMemcpyDeviceToDevice);
        float res_norm = 0.0f;
        int k = 0;
        do {
            exchange_boundaries(d_u_old, nx, ny_local, rank, size);
            jacobi_step<<<blocks, threads>>>(d_u_new, d_u_old, d_rhs, nx, ny_local, dx2, dy2, alpha_dt);
            enforce_neumann_bc(d_u_new, nx, ny_local, rank, size);

            compute_residual<<<blocks, threads>>>(d_u_new, d_u_old, d_res, nx, ny_local);
            cudaMemcpy(h_res, d_res, bytes, cudaMemcpyDeviceToHost);
            float local_sum = 0.0f;
            for (int i = nx; i < nx * (ny_local - 1); ++i) local_sum += h_res[i];
            float global_sum;
            MPI_Allreduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            res_norm = sqrt(global_sum);
            // if (rank == 0 && t % save_interval == 0 && k % 1 == 0)
            //     printf("t=%d iter=%d res=%.6e\n", t, k, res_norm);
            std::swap(d_u_old, d_u_new);
            k++;
        } while (res_norm > tol && k < max_jacobi_iters);

        if (t % save_interval == 0)
            save_solution(d_u_old, nx, ny_local, rank, size, t, ny_total);
    }

    // Record the end time
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Print the execution time
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;

    delete[] h_res;
    cudaFree(d_u_old); cudaFree(d_u_new); cudaFree(d_rhs); cudaFree(d_res);
    MPI_Finalize();
    return 0;
}
