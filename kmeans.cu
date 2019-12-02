#include <algorithm>
#include <cfloat>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
#include "cuda.h"
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

using namespace cv;

__device__ float
squared_l3_distance(float x_1, float y_1, float z_1, float x_2, float y_2, float z_2) {
  return (x_1 - x_2) * (x_1 - x_2)
       + (y_1 - y_2) * (y_1 - y_2)
       + (z_1 - z_2) * (z_1 - z_2);
}

__global__ void assign_clusters(const cv::cuda::PtrStepSzf data,
                                int row_size,
                                cv::cuda::PtrStepSzf means,
                                cv::cuda::PtrStepSzf new_sums,
                                int k,
                                cv::cuda::PtrStepSz<int32_t> counts) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= row_size) return;

  // Make global loads once.
  const float x = data(0, index);
  const float y = data(1, index);
  const float z = data(2, index);

  float best_distance = FLT_MAX;
  int best_cluster = 0;
  for (int cluster = 0; cluster < k; ++cluster) {
    const float distance = squared_l3_distance(x, y, z,
        means(0, cluster), means(1, cluster), means(2, cluster));
    if (distance < best_distance) {
      best_distance = distance;
      best_cluster = cluster;
    }
  }

  atomicAdd(&new_sums(0, best_cluster), x);
  atomicAdd(&new_sums(1, best_cluster), y);
  atomicAdd(&new_sums(2, best_cluster), z);
  atomicAdd(&counts(0, best_cluster), 1);
}

__global__ void compute_new_means_and_reset(cv::cuda::PtrStepSzf means,
                                            cv::cuda::PtrStepSzf new_sums,
                                            cv::cuda::PtrStepSz<int32_t> counts) {
  const int cluster = threadIdx.x;
  const int count = max(1, counts[cluster]);
  means(0, cluster) = new_sums(0, cluster) / count;
  means(1, cluster) = new_sums(1, cluster) / count;
  means(2, cluster) = new_sums(2, cluster) / count;

  new_sums(0, cluster) = 0;
  new_sums(1, cluster) = 0;
  new_sums(2, cluster) = 0;
  counts(0, cluster) = 0;
}

static Mat generate_random_means(Mat src, size_t k){
  Mat img = src.reshape(3, 1);
  img.convertTo(img, CV_32FC3);

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> distribution(0,img.cols);

  Mat centers(k, 1, CV_32FC3);
  for(int i = 0; i < k; ++i){
    centers.at<Vec3f>(i, 0) = img.at<Vec3f>(0, distribution(rng));
  }
  centers = centers.reshape(1, k);
  return centers.t();
}

static cv::cuda::GpuMat g_data, g_means, g_sums, g_counts;

Mat kmeans(Mat src, Mat means, size_t k, size_t max_iterations){
  Mat data;
  resize(src, data, src.size()/2); // Down-sample original image
  data = data.reshape(1, data.total());
  data = data.t();
  data.convertTo(data, CV_32F);
  g_data.upload(data);

  if(means.empty()){
    means = generate_random_means(src, k);
  }
  g_means.upload(means);

  if(g_sums.empty() || g_sums.size() != means.size()) {
    g_sums.create(means.size(), CV_32F);
  }
  g_sums.setTo(Scalar::all(0.0));

  if(g_counts.empty() || g_counts.size() != means.size()) {
    g_counts.create(means.size(), CV_32S);
  }
  g_counts.setTo(Scalar::all(0.0));

  const size_t threads = 1024;
  size_t number_of_elements = g_data.cols;
  int blocks = (number_of_elements + threads - 1) / threads;

  for (size_t iteration = 0; iteration < max_iterations; ++iteration) {
    assign_clusters <<<blocks, threads>>>(g_data, number_of_elements, g_means, g_sums, k, g_counts);
    cudaDeviceSynchronize();
    compute_new_means_and_reset<<<1, k>>>(g_means, g_sums, g_counts);
    cudaDeviceSynchronize();

    Mat new_means;
    g_means.download(new_means);
    if(norm(means, new_means) < 1.0){
      // Stop early if less than threshold
      break;
    }
    means = new_means;
  }
  return means;
}