#ifndef VRVISOR_EFFECTS_H
#define VRVISOR_EFFECTS_H

#define HALF_IMG_ROWS 240
#define NBHD_SIZE 16
#define NUM_THREADS 8

// # of threads should be HALF_IMG_ROWS / NBHD_SIZE
#define NUM_THREADS_HALFTONE 30

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
