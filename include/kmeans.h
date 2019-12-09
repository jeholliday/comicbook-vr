#ifndef VRVISOR_KMEANS_H
#define VRVISOR_KMEANS_H

#include "capture.h"

#include <opencv2/opencv.hpp>

extern cv::Mat kmeans(cv::Mat src, cv::Mat means, size_t k, size_t max_iterations);

static void* kmeans_thread(void* arg);

class Kmeans {
public:
    Kmeans(int k, int num_iterations, ImageCapture* src);

    ~Kmeans();

    cv::Mat getMeans();

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
