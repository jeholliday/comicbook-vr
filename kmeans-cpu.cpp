#include <opencv2/opencv.hpp>

cv::Mat kmeans(cv::Mat src, cv::Mat means, size_t k, size_t max_iterations){
  cv::Mat samples = src.reshape(1, src.total());
  samples.convertTo(samples, CV_32F);

  cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, max_iterations, 1.0);

  cv::Mat centers, labels;
  cv::kmeans(samples, k, labels, criteria, 1,cv::KMEANS_PP_CENTERS, centers);

  return centers.t();
}