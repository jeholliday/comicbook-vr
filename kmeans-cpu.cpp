#include <opencv2/opencv.hpp>

/**
 * CPU only implementation of kmeans algorithm
 * @param src Source Image
 * @param means Starting point for discrete colors
 * @param k Number of discrete colors
 * @param max_iterations Number of iterations to improve means
 * @return New color set
 */
cv::Mat kmeans(cv::Mat src, cv::Mat means, size_t k, size_t max_iterations)
{
    cv::Mat samples;
    cv::resize(src, samples, src.size() / 4);
    samples = samples.reshape(1, samples.rows * samples.cols);
    samples.convertTo(samples, CV_32F);

    cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, max_iterations, 1.0);

    cv::Mat centers, labels;
    cv::kmeans(samples, k, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers);

    return centers.t();
}