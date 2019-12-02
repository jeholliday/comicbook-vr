#ifndef VRVISOR_PIPELINE_H
#define VRVISOR_PIPELINE_H

#include <opencv2/opencv.hpp>
#include "capture.h"
#include "kmeans.h"

static cv::Mat process_image(cv::Mat src);

static void* pipeline_thread(void* arg);

class Pipeline {
public:
  Pipeline(ImageCapture* capture, Kmeans* kmeans_src);

  ~Pipeline();

  void start();

  cv::Mat join();

private:
  ImageCapture* capture;
  Kmeans* kmeans_src;
  cv::Mat result;
  int last_frame;

  pthread_t thread;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  bool running;

  friend void* pipeline_thread(void* arg);
};

#endif // VRVISOR_PIPELINE_H
