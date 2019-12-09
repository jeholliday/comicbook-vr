#include "effects.h"
#include "kmeans.h"
#include "timing.h"

#include <opencv2/opencv.hpp>

/**
 * offline.cpp
 * Process single image into comicbook image.
 */

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "usage: <data-file> [k] [iterations] " << std::endl;
        std::exit(EXIT_FAILURE);
    }
    int k = 8, iterations = 100;
    if (argc > 2) {
        k = std::atoi(argv[2]);
    }
    if (argc > 3) {
        iterations = std::atoi(argv[3]);
    }

    Mat image = imread(argv[1]);
    resize(image, image, Size(640, 480));

    START_TIMING();
    try {
        Mat canny_overlay = Effects::canny(image);
        Mat means = kmeans(image, Mat(), k, iterations);
        Mat posterized = Effects::posterize(image, means);
        Mat halftone_overlay = Effects::halftone(image);
        Mat combined = Effects::overlay(canny_overlay, halftone_overlay, posterized);
        imwrite("post.jpg", combined);
    } catch (Exception& e) {
        std::cout << e.what() << std::endl;
    } catch (...) {
        std::cout << "Caught unexpected exception!" << std::endl;
    }
    STOP_TIMING("Frame Time")
    return 0;
}