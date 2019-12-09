#include "capture.h"
#include "effects.h"
#include "kmeans.h"
#include "timing.h"

#include <csignal>
#include <opencv2/opencv.hpp>

/**
 * live.cpp
 * Process images in realtime from a single camera.
 */

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

    ImageCapture capture(0);
    Kmeans kmeans_src(8, 100, &capture);
    size_t last_frame = 0;

    // Make window show up fullscreen
    namedWindow("Window", cv::WINDOW_NORMAL);
    setWindowProperty("Window", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    while (!stop) {
        START_TIMING();
        try {
            struct Frame frame = capture.getFrame(last_frame);
            Mat image = frame.image;
            resize(image, image, image.size() / 2);

            Mat canny_overlay = Effects::canny(image);
            Mat posterized = Effects::posterize(image, kmeans_src.getMeans());
            Mat halftone_overlay = Effects::halftone(image);
            Mat combined = Effects::overlay(canny_overlay, halftone_overlay, posterized);

            resize(combined, combined, combined.size() * 2);
            imshow("Window", combined);
        } catch (Exception e) {
            std::cout << e.what() << std::endl;
            break;
        } catch (...) {
            std::cout << "Caught unexpected exception!" << std::endl;
            break;
        }
        STOP_TIMING("Frame Time");

        if (waitKey(1) == 27)
            break; // stop capturing by pressing ESC
    }
    capture.stop();
    kmeans_src.stop();

    return 0;
}