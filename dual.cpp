#include "capture.h"
#include "kmeans.h"
#include "pipeline.h"
#include "timing.h"

#include <csignal>
#include <opencv2/opencv.hpp>

/**
 * dual.cpp
 * Process images from two cameras at the same time using the pipeline class.
 */

using namespace cv;

bool stop = false;

void stop_handler(int s) { stop = true; }

int main(int argc, char** argv)
{
    // Catch any stop signals to ensure proper cleanup
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = stop_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    ImageCapture left_cap(0);
    ImageCapture right_cap(1);

    Kmeans kmeans_src(8, 100, &left_cap);

    Pipeline left_pipeline(&left_cap, &kmeans_src);
    Pipeline right_pipeline(&right_cap, &kmeans_src);

    // Make window show up fullscreen
    namedWindow("Window", cv::WINDOW_NORMAL);
    setWindowProperty("Window", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    while (!stop) {
        START_TIMING();
        try {
            left_pipeline.start();
            right_pipeline.start();

            Mat left_image = left_pipeline.join();
            Mat right_image = right_pipeline.join();

            Mat array[] = { left_image, right_image };

            Mat final;
            cv::hconcat(array, 2, final);
            imshow("Window", final);
        } catch (Exception& e) {
            std::cout << e.what() << std::endl;
            break;
        } catch (...) {
            break;
        }
        STOP_TIMING("Frame Time");

        if (waitKey(1) == 27)
            break; // stop capturing by pressing ESC
    }
    left_cap.stop();
    right_cap.stop();
    kmeans_src.stop();

    return 0;
}