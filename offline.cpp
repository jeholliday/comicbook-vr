#include "capture.h"
#include "effects.h"
#include "kmeans.h"

#include <csignal>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv)
{
  if (argc < 2) {
    std::cerr << "usage: <data-file> [k] [iterations] "
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  int k = 8, iterations = 100;
  if(argc > 2) {
    k = std::atoi(argv[2]);
  }
  if(argc > 3) {
    iterations = std::atoi(argv[3]);
  }

  Mat image = imread(argv[1]);

  auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
  try {
    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    Mat c = Effects::canny(image);
    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    std::cout << "Canny: " << (end-start).count() << " ms" << std::endl;

    start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    Mat means = kmeans(image, Mat(), k, iterations);
    end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    std::cout << "kmeans: " << (end-start).count() << " ms" << std::endl;

    start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    Mat posterized = Effects::posterize(image, means);
    end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    std::cout << "posterized: " << (end-start).count() << " ms" << std::endl;

    start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    posterized = Effects::blur(posterized);
    end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    std::cout << "blur: " << (end-start).count() << " ms" << std::endl;

    start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    Mat combined = Effects::canny_overlay(c, posterized);
    end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    std::cout << "overlay: " << (end-start).count() << " ms" << std::endl;

    start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    Effects::halftone(image, combined);
    end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    std::cout << "halftone: " << (end-start).count() << " ms" << std::endl;

    imwrite("post.jpg", combined);
  }catch(Exception e){
    std::cout << e.what() << std::endl;
  }catch(...){

  }

  auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
  std::cerr << "Frame time: " << (end - start).count() << std::endl << std::endl;

  return 0;
}