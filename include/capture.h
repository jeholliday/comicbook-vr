#ifndef VRVISOR_CAPTURE_H
#define VRVISOR_CAPTURE_H

#include <opencv2/opencv.hpp>

/**
 * Internal thread for ImageCapture to continuous capture images
 * @param arg ImageCapture* to parent object
 * @return NULL
 */
static void* capture_thread(void* arg);

/**
 * Struct for returning a Mat with a unique frame number
 */
struct Frame {
    cv::Mat image;
    size_t frame_num;
};

/**
 * Object for continuously taking images from a VideoCapture
 *      and returning the latest image without blocking.
 */
class ImageCapture {
public:
    /**
     * Construct with a given camera id
     * @param id Camera id
     */
    ImageCapture(int id);

    ~ImageCapture();

    /**
     * Return the latest frame. Only block if a new frame isn't available.
     * @param lastFrame Frame id of last image returned
     * @return Next frame
     */
    struct Frame getFrame(size_t lastFrame);

    /**
     * Stop internal thread and free VideoCapture
     */
    void stop();

private:
    pthread_t thread;

protected:
    cv::VideoCapture cap;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    cv::Mat image;
    size_t frame_num;
    bool stopped;

    friend void* capture_thread(void* arg);
};

#endif // VRVISOR_CAPTURE_H
