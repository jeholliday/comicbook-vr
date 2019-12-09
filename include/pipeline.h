#ifndef VRVISOR_PIPELINE_H
#define VRVISOR_PIPELINE_H

#include "capture.h"
#include "kmeans.h"
#include <opencv2/opencv.hpp>

static cv::Mat process_image(cv::Mat src);

static void* pipeline_thread(void* arg);

struct canny_thread_args {
    cv::Mat* src;
    cv::Mat* result;
};

/**
 * Thread for performing Canny edge detection asynchronously
 * @param arg canny_thread_args*
 * @return NULL
 */
static void* canny_thread(void* arg);

struct halftone_thread_args {
    cv::Mat* src;
    cv::Mat* result;
};

/**
 * Thread for performing Halftone asynchronously
 * @param arg halftone_thread_args*
 * @return NULL
 */
static void* halftone_thread(void* arg);

struct posterized_thread_args {
    cv::Mat* src;
    cv::Mat* centers;
    cv::Mat* result;
};

/**
 * Thread for performing posterize asynchronously
 * @param arg posterized_thread_args*
 * @return NULL
 */
static void* posterized_thread(void* arg);

/**
 * Process image with Canny, Halftone, and Posterize all asynchronously
 * @param src Source Image
 * @param means Discrete colors
 * @return Comicbook image
 */
static cv::Mat process_image(cv::Mat src, cv::Mat means);

/**
 * Thread for processing an image asynchronously
 * @param arg Pipeline* to parent object
 * @return NULL
 */
static void* pipeline_thread(void* arg);

/**
 * Object for processing an image from an ImageCapture using an image pipeline
 */
class Pipeline {
public:
    Pipeline(ImageCapture* capture, Kmeans* kmeans_src);

    ~Pipeline();

    /**
     * Take latest image and means and process an image
     */
    void start();

    /**
     * Wait for latest processing image
     * @return Processed image
     */
    cv::Mat join();

private:
    ImageCapture* capture;
    Kmeans* kmeans_src;
    cv::Mat result;
    int last_frame;

    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    bool running;

    friend void* pipeline_thread(void* arg);
};

#endif // VRVISOR_PIPELINE_H
