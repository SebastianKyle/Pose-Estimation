#include "utils.cpp"

__device__ void clamp(int& pos, int max_pos) {
  pos = pos > 0 ? pos : 0;
  pos = pos < max_pos ? pos : max_pos - 1;
}

struct WeightIdx {
    float weight;
    int idx;
};

__global__ void decode_simcc_kernel(
    const float* simcc_labels,
    int size,
    int label_size,
    int num_joints, 
    float* out
) {
    extern __shared__ WeightIdx sh_kernel_data[];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    int label_idx = idx % label_size;
    int joint_idx = (idx / label_size) % num_joints;
    int batch_idx = idx / (num_joints * label_size);

    if (idx >= size)
        return;

    sh_kernel_data[tid] = { simcc_labels[idx], label_idx };
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sh_kernel_data[tid + s].weight > sh_kernel_data[tid].weight) {
                sh_kernel_data[tid].weight = sh_kernel_data[tid + s].weight;
                sh_kernel_data[tid].idx = sh_kernel_data[tid + s].idx;
            } 
        }

        __syncthreads();
    }

    if (tid == 0) {
        out[batch_idx * num_joints + joint_idx] = (float) sh_kernel_data[0].idx / (float) label_size;
    }
}

__global__ void softmax_kernel(
    const float* input,
    const float* input_arg_max,
    int size,
    int label_size,
    int batch_size,
    int num_joints,
    float beta,
    float sigma,
    float* output
) {
    extern __shared__ float sh_softmax_data[];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    int joint_idx = (idx / label_size) % num_joints;
    int batch_idx = idx / (num_joints * label_size);

    if (idx >= size)
        return;

    int arg_max = batch_idx * num_joints * label_size + joint_idx * label_size + (int) (input_arg_max[batch_idx * num_joints + joint_idx] * (float) label_size);
    sh_softmax_data[tid] = expf((input[idx] - input[arg_max]) * beta * sigma);
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_softmax_data[tid] += sh_softmax_data[tid + s];
        }
        __syncthreads();
    }

    output[idx] = expf((input[idx] - input[arg_max]) * beta * sigma) / sh_softmax_data[0];
}

__global__ void maximum_kernel(
    const float* input,
    int size,
    int label_size,
    int batch_size,
    int num_joints,
    float* output
) {
    extern __shared__ float sh_maximum_data[];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    int joint_idx = (idx / label_size) % num_joints;
    int batch_idx = idx / (num_joints * label_size);

    if (idx >= size)
        return;

    sh_maximum_data[tid] = input[idx];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_maximum_data[tid] = max(sh_maximum_data[tid], sh_maximum_data[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[batch_idx * num_joints + joint_idx] = sh_maximum_data[tid];
    }
}

__global__ void gaussian_blur_1d(
    const float* input,
    const float* org_arg_max,
    int size,
    int label_size,
    int num_joints,
    int kernel_size,
    float sigma,
    float* output
) {
    extern __shared__ float sh_gaussian_data[];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    int joint_idx = (idx / label_size) % num_joints;
    int batch_idx = idx / (num_joints * label_size);

    if (idx > size)
        return;

    sh_gaussian_data[tid] = input[idx];
    __syncthreads();

    int radius = kernel_size / 2;

    float blur_res = 0.0f;
    float sum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        int neighbor_idx = tid + i;
        clamp(neighbor_idx, label_size);

        float gau_weight = expf( -( i * i ) / ( 2.0f * sigma * sigma ) );
        blur_res += sh_gaussian_data[neighbor_idx] * gau_weight;
        sum += gau_weight;
    }
    __syncthreads();

    int max_idx = (int) (org_arg_max[batch_idx * num_joints + joint_idx] * (float) label_size);
    output[idx] = (blur_res / sum) * (input[max_idx] / sh_gaussian_data[max_idx]);
}

__global__ void dark_refine(
    const float* simcc_labels,
    float* refined_pos,
    int label_size,
    int size,
    int batch_size,
    int num_joints
) {
    extern __shared__ float sh_dark_data[];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    int joint_idx = (idx / label_size) % num_joints;
    int batch_idx = idx / (num_joints * label_size);

    if (idx > size)
        return;

    sh_dark_data[tid] = simcc_labels[idx];
    __syncthreads();

    int pos = (int) (refined_pos[batch_idx * num_joints + joint_idx] * (float) label_size);

    int pos1 = pos + 1;
    clamp(pos1, label_size);
    int pos_1 = pos - 1;
    clamp(pos_1, label_size);
    int pos2 = pos + 2;
    clamp(pos2, label_size);
    int pos_2 = pos - 2;
    clamp(pos_2, label_size);

    float dx0 = sh_dark_data[pos];
    float dx1 = sh_dark_data[pos1];
    float dx_1 = sh_dark_data[pos_1];
    float dx2 = sh_dark_data[pos2];
    float dx_2 = sh_dark_data[pos_2];

    float dx = 0.5f * (dx1 - dx_1);
    float dxx = 1e-9f + 0.25 * (dx2 - 2 * dx0 + dx_2);

    float offset = dx / dxx;

    float new_pos = refined_pos[batch_idx * num_joints + joint_idx] * (float) label_size + offset;

    refined_pos[batch_idx * num_joints + joint_idx] = new_pos / (float) label_size;
}

void decode_joints(
    const float* d_x_labels,
    const float* d_y_labels,
    float* d_out_x,
    float* d_out_y,
    int batch_size,
    int num_joints,
    int W,
    int H,
    int x_size,
    int y_size
) {
    int threads_per_block_x = W;
    int blocks_per_grid_x = batch_size * num_joints;
    decode_simcc_kernel<<< blocks_per_grid_x, threads_per_block_x, threads_per_block_x * (sizeof(float) + sizeof(int)) >>>(d_x_labels, x_size, W, num_joints, d_out_x);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError()); 

    int threads_per_block_y = H;
    int blocks_per_grid_y = batch_size * num_joints;
    decode_simcc_kernel<<< blocks_per_grid_y, threads_per_block_y, threads_per_block_y * (sizeof(float) + sizeof(int)) >>>(d_y_labels, y_size, H, num_joints, d_out_y);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void refine_joints(
    const float* d_x_labels,
    const float* d_y_labels,
    float* d_out_x,
    float* d_out_y,
    float* d_x_blur,
    float* d_y_blur,
    int batch_size,
    int num_joints,
    int W,
    int H,
    int x_size,
    int y_size,
    float sigma
) {
    int threads_per_block_x = W;
    int blocks_per_grid_x = batch_size * num_joints;

    int threads_per_block_y = H;
    int blocks_per_grid_y = batch_size * num_joints;

    int x_blur_kernel = (int) (sigma * 20 - 7) / 3;
    gaussian_blur_1d<<< blocks_per_grid_x, threads_per_block_x, threads_per_block_x * sizeof(float) >>>(
        d_x_labels,
        d_out_x,
        x_size,
        W,
        num_joints,
        x_blur_kernel,
        sigma,
        d_x_blur
    );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    int y_blur_kernel = (int) (sigma * 20 - 7) / 3;
    gaussian_blur_1d<<< blocks_per_grid_y, threads_per_block_y, threads_per_block_y * sizeof(float) >>>(
        d_y_labels,
        d_out_y,
        y_size,
        H,
        num_joints,
        y_blur_kernel,
        sigma,
        d_y_blur
    );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    dark_refine<<< blocks_per_grid_x, threads_per_block_x, threads_per_block_x * sizeof(float) >>>(
        d_x_blur,
        d_out_x,
        W,
        x_size,
        batch_size,
        num_joints
    );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    dark_refine<<< blocks_per_grid_y, threads_per_block_y, threads_per_block_y * sizeof(float) >>>(
        d_y_blur,
        d_out_y,
        H,
        y_size,
        batch_size,
        num_joints
    );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void decode_visibility(
    const float* d_x_labels,
    const float* d_y_labels,
    const float* d_out_x,
    const float* d_out_y,
    float* d_x_softmax,
    float* d_y_softmax,
    float* d_x_visibility,
    float* d_y_visibility,
    int batch_size,
    int num_joints,
    int W,
    int H,
    int x_size,
    int y_size,
    float sigma,
    float decode_beta
) {
    int threads_per_block_x = W;
    int blocks_per_grid_x = batch_size * num_joints;

    int threads_per_block_y = H;
    int blocks_per_grid_y = batch_size * num_joints;

    softmax_kernel<<< blocks_per_grid_x, threads_per_block_x, threads_per_block_x * sizeof(float) >>>(
        d_x_labels, 
        d_out_x,
        x_size, 
        W, 
        batch_size, 
        num_joints, 
        decode_beta, 
        sigma,
        d_x_softmax
    );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    softmax_kernel<<< blocks_per_grid_y, threads_per_block_y, threads_per_block_y * sizeof(float) >>>(
        d_y_labels, 
        d_out_y,
        y_size, 
        H, 
        batch_size, 
        num_joints, 
        decode_beta, 
        sigma,
        d_y_softmax
    );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    maximum_kernel<<< blocks_per_grid_x, threads_per_block_x, threads_per_block_x * sizeof(float) >>>(
        d_x_softmax, 
        x_size, 
        W, 
        batch_size, 
        num_joints, 
        d_x_visibility
    );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    maximum_kernel<<< blocks_per_grid_y, threads_per_block_y, threads_per_block_y * sizeof(float) >>>(
        d_y_softmax, 
        y_size, 
        H, 
        batch_size, 
        num_joints, 
        d_y_visibility
    );
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void decode_simcc_labels(
    const std::vector<cv::Mat>& x_labels, 
    const std::vector<cv::Mat>& y_labels, 
    int batch_size,
    int num_joints,
    int W,
    int H,
    float sigma,
    float decode_beta,
    std::vector<cv::Point2f>& joints,
    std::vector<float>& visibility
) {
    int x_size = batch_size * num_joints * W * (int) sizeof(float);
    int y_size = batch_size * num_joints * H * (int) sizeof(float);

    float* d_x_labels;
    float* d_y_labels;
    float* d_out_x;
    float* d_out_y;

    // Decode joints locations
    checkCudaErrors(cudaMalloc(&d_x_labels, x_size));
    checkCudaErrors(cudaMalloc(&d_y_labels, y_size));
    checkCudaErrors(cudaMalloc(&d_out_x, batch_size * num_joints * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_out_y, batch_size * num_joints * sizeof(float)));

    for (int i = 0; i < batch_size; ++i) {
        checkCudaErrors(cudaMemcpy(d_x_labels + i * num_joints * W, x_labels[i].ptr<float>(), num_joints * W * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_y_labels + i * num_joints * H, y_labels[i].ptr<float>(), num_joints * H * sizeof(float), cudaMemcpyHostToDevice));
    }

    decode_joints(
        d_x_labels, d_y_labels,
        d_out_x, d_out_y,
        batch_size,
        num_joints,
        W, H,
        x_size, y_size
    );

    // DARK refinement
    float* d_x_blur;
    float* d_y_blur;

    checkCudaErrors(cudaMalloc(&d_x_blur, x_size));
    checkCudaErrors(cudaMalloc(&d_y_blur, y_size));

    refine_joints(
        d_x_labels, d_y_labels,
        d_out_x, d_out_y,
        d_x_blur, d_y_blur,
        batch_size,
        num_joints,
        W, H,
        x_size, y_size,
        sigma
    );

    std::vector< float > decoded_x(batch_size * num_joints);
    std::vector< float > decoded_y(batch_size * num_joints);
    checkCudaErrors(cudaMemcpy(decoded_x.data(), d_out_x, batch_size * num_joints * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(decoded_y.data(), d_out_y, batch_size * num_joints * sizeof(float), cudaMemcpyDeviceToHost));

    joints.resize(batch_size * num_joints);
    for (int i = 0; i < batch_size * num_joints; ++i) {
        joints[i] = cv::Point2f(decoded_x[i], decoded_y[i]);
    }

    // Decode visibility
    float* d_x_softmax;
    float* d_y_softmax;
    checkCudaErrors(cudaMalloc(&d_x_softmax, x_size));
    checkCudaErrors(cudaMalloc(&d_y_softmax, y_size));

    float* d_x_visibility;
    float* d_y_visibility;
    checkCudaErrors(cudaMalloc(&d_x_visibility, batch_size * num_joints * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_y_visibility, batch_size * num_joints * sizeof(float)));

    decode_visibility(
        d_x_labels, d_y_labels,
        d_out_x, d_out_y,
        d_x_softmax, d_y_softmax,
        d_x_visibility, d_y_visibility,
        batch_size,
        num_joints,
        W, H,
        x_size, y_size,
        sigma,
        decode_beta
    );

    std::vector< float > x_visibility(batch_size * num_joints);
    std::vector< float > y_visibility(batch_size * num_joints);
    checkCudaErrors(cudaMemcpy(x_visibility.data(), d_x_visibility, batch_size * num_joints * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(y_visibility.data(), d_y_visibility, batch_size * num_joints * sizeof(float), cudaMemcpyDeviceToHost));

    visibility.resize(batch_size * num_joints);
    for (int i = 0; i < batch_size * num_joints; ++i) {
        visibility[i] = min(x_visibility[i], y_visibility[i]);
    }

    cudaFree(d_x_labels);
    cudaFree(d_y_labels);
    cudaFree(d_out_x);
    cudaFree(d_out_y);
    cudaFree(d_x_blur);
    cudaFree(d_y_blur);
    cudaFree(d_x_softmax);
    cudaFree(d_y_softmax);
    cudaFree(d_x_visibility);
    cudaFree(d_y_visibility);
}