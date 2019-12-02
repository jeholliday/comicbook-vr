#ifndef VRVISOR_EFFECTS_H
#define VRVISOR_EFFECTS_H

#define NUM_THREADS 8

#include <opencv2/opencv.hpp>

using namespace cv;

class Effects {
public:
  static Mat canny(Mat src);

  static Mat blur(Mat src);

  static Mat posterize(Mat src, Mat centers);

  static void* posterize_thread(void* arg);

  static Mat halftone(Mat src);

  static Mat overlay(Mat canny_overlay, Mat posterized_image, Mat halftone_overlay);

  static void* halftone_thread(void* arg);

  struct posterize_args{
    int start_index;
    int end_index;
    Mat* img;
    Mat* centers;
    Mat* new_image;
  };

  struct halftone_args{
      int start_index;
      int end_index;
      Mat* img;
      Mat* gray_img;
      Mat* new_image;
  };

  static const int brightness = 255;

};

#endif // VRVISOR_EFFECTS_H
