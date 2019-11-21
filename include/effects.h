#ifndef VRVISOR_EFFECTS_H
#define VRVISOR_EFFECTS_H

#define NUM_THREADS 8

#include <opencv2/opencv.hpp>

using namespace cv;

class Effects {
public:
  static Mat canny(Mat src);

  static Mat canny_overlay(Mat alpha, Mat back);

  static Mat blur(Mat src);

  static Mat posterize(Mat src, Mat centers);

  static void* posterize_thread(void* arg);

  static Mat halftone(Mat src, Mat output);

  struct posterize_args{
    int start_index;
    int end_index;
    Mat* img;
    Mat* centers;
    Mat* new_image;
  };

  static const int brightness = 255;

};

#endif // VRVISOR_EFFECTS_H
