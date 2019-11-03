#ifndef VRVISOR_CAPTURE_H
#define VRVISOR_CAPTURE_H

#include <opencv2/opencv.hpp>

static void* capture_thread(void* arg);

struct Frame {
  cv::Mat image;
  size_t frame_num;
};

class ImageCapture{
public:
  ImageCapture(int id);

  ~ImageCapture();

  struct Frame getFrame(size_t lastFrame);

  void stop();

private:
  pthread_t thread;

protected:
  cv::VideoCapture cap;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  cv::Mat image;
  size_t frame_num;
  bool stopped;

  friend void* capture_thread(void* arg);
};

#endif // VRVISOR_CAPTURE_H
