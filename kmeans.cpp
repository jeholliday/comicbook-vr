#include "kmeans.h"

/**
 * Thread to continuously calculate color set
 * @param arg Kmeans* to parent object
 * @return NULL
 */
void* kmeans_thread(void* arg)
{
    Kmeans* parent = (Kmeans*)arg;

    size_t last_frame = 0;
    while (!parent->stopped) {
        struct Frame frame = parent->src->getFrame(last_frame);
        last_frame = frame.frame_num;
        cv::Mat new_means = kmeans(frame.image, parent->means, parent->k, parent->num_iterations);

        // Lock mutex before copying latest means to Kmeans object
        pthread_mutex_lock(&(parent->mutex));
        new_means.copyTo(parent->means);
        pthread_cond_broadcast(&(parent->cond));
        pthread_mutex_unlock(&(parent->mutex));
    }
}

/**
 * Get latest calculated means
 * @return Latest means
 */
Kmeans::Kmeans(int k, int num_iterations, ImageCapture* src)
    : k(k)
    , num_iterations(num_iterations)
    , src(src)
    , stopped(false)
{
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond, NULL);
    pthread_create(&thread, NULL, &kmeans_thread, this);
}

Kmeans::~Kmeans() { stop(); }

/**
 * Get latest calculated means
 * @return Latest means
 */
cv::Mat Kmeans::getMeans()
{
    pthread_mutex_lock(&mutex);
    while (means.empty()) {
        pthread_cond_wait(&cond, &mutex);
    }
    cv::Mat latest;
    means.copyTo(latest);
    pthread_mutex_unlock(&mutex);
    return latest;
}

/**
 * Stop internal thread
 */
void Kmeans::stop()
{
    pthread_mutex_lock(&mutex);
    if (!stopped) {
        stopped = true;
        pthread_mutex_unlock(&mutex);
        pthread_join(thread, NULL);
        pthread_cond_broadcast(&cond); // Wakeup anybody waiting
    } else {
        pthread_mutex_unlock(&mutex);
    }
}
