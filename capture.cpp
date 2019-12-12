#include "capture.h"

/**
 * Internal thread for ImageCapture to continuous capture images
 * @param arg ImageCapture* to parent object
 * @return NULL
 */
void* capture_thread(void* arg)
{
    ImageCapture* capture = (ImageCapture*)arg;
    pthread_mutex_lock(&(capture->mutex));
    bool stop = capture->stopped;
    pthread_mutex_unlock(&(capture->mutex));

    // Continuously process until stopped
    while (!stop) {
        cv::Mat image;
        capture->cap >> image;

        cv::resize(image, image, cv::Size(640, 480));
        cv::flip(image, image, -1);

        // Lock mutex before copying new frame to ImageCapture
        pthread_mutex_lock(&(capture->mutex));
        image.copyTo(capture->image);
        capture->frame_num += 1;
        stop = capture->stopped;
        pthread_cond_broadcast(&(capture->cond)); // Signal any waiting threads
        pthread_mutex_unlock(&(capture->mutex));
    }
    return NULL;
}

/**
 * Construct with a given camera id
 * @param id Camera id
 */
ImageCapture::ImageCapture(int id)
    : cap(id)
    , frame_num(0)
    , stopped(false)
{
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond, NULL);
    pthread_create(&thread, NULL, &capture_thread, this);

    // Set camera image capture FPS
    cap.set(cv::CAP_PROP_FPS, 30);
}

ImageCapture::~ImageCapture() { stop(); }

/**
 * Return the latest frame. Only block if a new frame isn't available.
 * @param lastFrame Frame id of last image returned
 * @return Next frame
 */
struct Frame ImageCapture::getFrame(size_t last_frame)
{
    pthread_mutex_lock(&mutex);
    // Block if current frame matches last_frame
    while (last_frame == frame_num) {
        pthread_cond_wait(&cond, &mutex);
    }
    cv::Mat latest;
    image.copyTo(latest);
    size_t frame = frame_num;
    pthread_mutex_unlock(&mutex);
    return { latest, frame };
}

/**
 * Stop internal thread and free VideoCapture
 */
void ImageCapture::stop()
{
    pthread_mutex_lock(&mutex);
    if (!stopped) {
        stopped = true;
        pthread_mutex_unlock(&mutex);
        pthread_join(thread, NULL);
        pthread_mutex_lock(&mutex);
        cap.release();
        frame_num += 1;
        pthread_cond_broadcast(&cond); // Wakeup anybody waiting
    }
    pthread_mutex_unlock(&mutex);
}