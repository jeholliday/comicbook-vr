#ifndef VRVISOR_KMEANS_H
#define VRVISOR_KMEANS_H

#include "capture.h"

#include <opencv2/opencv.hpp>

/**
 * Forward definition of kmeans algorithm to allow multiple implementations
 * @param src Source Image
 * @param means Starting point for discrete colors
 * @param k Number of discrete colors
 * @param max_iterations Number of iterations to improve means
 * @return New color set
 */
extern cv::Mat kmeans(cv::Mat src, cv::Mat means, size_t k, size_t max_iterations);

/**
 * Thread to continuously calculate color set
 * @param arg Kmeans* to parent object
 * @return NULL
 */
static void* kmeans_thread(void* arg);

class Kmeans {
public:
    Kmeans(int k, int num_iterations, ImageCapture* src);

    ~Kmeans();

    /**
     * Get latest calculated means
     * @return Latest means
     */
    cv::Mat getMeans();

    /**
     * Stop internal thread
     */
    void stop();

private:
    pthread_t thread;

protected:
    const int k;
    const int num_iterations;
    ImageCapture* src;

    pthread_mutex_t mutex;
    pthread_cond_t cond;
    cv::Mat means;
    bool stopped;

    friend void* kmeans_thread(void* arg);
};

#endif // VRVISOR_KMEANS_H
