#include "capture.h"
#include "effects.h"
#include "kmeans.h"
#include "pipeline.h"

#include <csignal>
#include <opencv2/opencv.hpp>

bool stop = false;

void stop_handler(int s) { stop = true; }

int main(int argc, char **argv) {

  struct sigaction sigIntHandler;

  sigIntHandler.sa_handler = stop_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;

  sigaction(SIGINT, &sigIntHandler, NULL);

  ImageCapture left_cap(0);
  ImageCapture right_cap(1);

  Kmeans kmeans_src(8, 100, &left_cap);

  namedWindow("Name", cv::WINDOW_NORMAL);
  setWindowProperty("Name", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

  Pipeline left_pipeline(&left_cap, &kmeans_src);
  Pipeline right_pipeline(&right_cap, &kmeans_src);

  while (!stop) {
    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch());

    try {
      left_pipeline.start();
      right_pipeline.start();

      Mat left_image = left_pipeline.join();
      Mat right_image = right_pipeline.join();

      Mat array[] = {left_image, right_image};

      Mat final;
      cv::hconcat(array, 2, final);

      imshow("Name", final);
    } catch (Exception &e) {
      std::cout << e.what() << std::endl;
      break;
    } catch (...) {
      break;
    }

    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch());

    std::cerr << "Frame time: " << (end - start).count() << std::endl
              << std::endl;

    if (waitKey(1) == 27)
      break; // stop capturing by pressing ESC
  }
  left_cap.stop();
  right_cap.stop();
  kmeans_src.stop();

  return 0;
}