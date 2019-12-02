#include "capture.h"
#include "effects.h"
#include "kmeans.h"
#include <chrono>

#include <csignal>
#include <opencv2/opencv.hpp>

bool stop = false;

void stop_handler(int s){
  stop = true;
}

int main(int argc, char** argv)
{

  struct sigaction sigIntHandler;

  sigIntHandler.sa_handler = stop_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;

  sigaction(SIGINT, &sigIntHandler, NULL);

  ImageCapture capture(0);
  Kmeans kmeans_src(8, 100, &capture);
  size_t last_frame = 0;

  namedWindow("Name", cv::WINDOW_NORMAL);
  setWindowProperty("Name", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

  while(!stop)
  {
    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    try {
      auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
      struct Frame frame = capture.getFrame(last_frame);
      Mat image = frame.image;
      auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
      std::cout << "Get image: " << (end-start).count() << " ms" << std::endl;

      start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
      resize(image, image, image.size()/2);
      end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
      std::cout << "Shrink: " << (end-start).count() << " ms" << std::endl;

      start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
      Mat c = Effects::canny(image);
      end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
      std::cout << "Canny: " << (end-start).count() << " ms" << std::endl;

      start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
      Mat posterized = Effects::posterize(image, kmeans_src.getMeans());
      end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
      std::cout << "posterized: " << (end-start).count() << " ms" << std::endl;

      start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
      posterized = Effects::blur(posterized);
      end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
      std::cout << "blur: " << (end-start).count() << " ms" << std::endl;

      start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
      Mat ht = Effects::halftone(image, posterized);
      end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
      std::cout << "halftone: " << (end-start).count() << " ms" << std::endl;

      start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
      Mat combined = Effects::canny_overlay(c, posterized, ht);
      end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
      std::cout << "overlay: " << (end-start).count() << " ms" << std::endl;

      start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
      resize(combined, combined, combined.size()*2);
      end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
      std::cout << "Grow: " << (end-start).count() << " ms" << std::endl;

      imshow("Name", combined);
    }catch(Exception e){
      std::cout << e.what() << std::endl;
      break;
    }catch(...){
      break;
    }

    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

    std::cerr << "Frame time: " << (end - start).count() << std::endl << std::endl;

    if( waitKey(1) == 27 ) break; // stop capturing by pressing ESC
  }
  capture.stop();
  kmeans_src.stop();

  return 0;
}