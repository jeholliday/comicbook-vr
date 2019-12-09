#ifndef VRVISOR_EFFECTS_H
#define VRVISOR_EFFECTS_H

#define NBHD_SIZE 9
#define NUM_THREADS 4

#include <opencv2/opencv.hpp>

using namespace cv;

class Effects {
public:
    /**
     * Perform canny edge detection
     * @param src Source Image
     * @return Mat with detected edges
     */
    static Mat canny(Mat src);

    /**
     * Helper method for blurring an image
     * @param src Source Image
     * @return Blurred Image
     */
    static Mat blur(Mat src);

    /**
     * Overlay Canny edges and halftone dots onto a posterized image
     * @return Combined image
     */
    static Mat overlay(Mat canny_overlay, Mat halftone_overlay, Mat posterized_image);

    /**
     * Color an image using a reduced color set
     * @param src Source image
     * @param centers Discrete colors
     * @return Posterized image
     */
    static Mat posterize(Mat src, Mat centers);

    // Struct to pass arguments to posterize_thread
    struct posterize_args {
        int start_index;
        int end_index;
        Mat* img;
        Mat* centers;
        Mat* new_image;
    };

    /**
     * Thread to posterize a subset of an image
     * @param arg posterize_args*
     * @return NULL
     */
    static void* posterize_thread(void* arg);

    /**
     * Create halftone effect from an image
     * @param src Source image
     * @return Image with halftone dots
     */
    static Mat halftone(Mat src);

    // Struct to pass argument to halftone_thread
    struct halftone_args {
        int start_index;
        int end_index;
        Mat* src;
        Mat* gray_img;
        Mat* new_image;
    };

    /**
     * Thread to perform halftone on a subset of an image
     * @param arg halftone_args*
     * @return NULL
     */
    static void* halftone_thread(void* arg);
};

#endif // VRVISOR_EFFECTS_H
