#include "pipeline.h"
#include "effects.h"

/**
 * Thread for performing Canny edge detection asynchronously
 * @param arg canny_thread_args*
 * @return NULL
 */
static void* canny_thread(void* arg)
{
    auto args = (struct canny_thread_args*)arg;
    *(args->result) = Effects::canny(*(args->src));
    return NULL;
}

/**
 * Thread for performing Halftone asynchronously
 * @param arg halftone_thread_args*
 * @return NULL
 */
static void* halftone_thread(void* arg)
{
    auto args = (struct halftone_thread_args*)arg;
    *(args->result) = Effects::halftone(*(args->src));
    return NULL;
}

/**
 * Thread for performing posterize asynchronously
 * @param arg posterized_thread_args*
 * @return NULL
 */
static void* posterized_thread(void* arg)
{
    auto args = (struct posterized_thread_args*)arg;
    *(args->result) = Effects::posterize(*(args->src), *(args->centers));
    return NULL;
}

/**
 * Process image with Canny, Halftone, and Posterize all asynchronously
 * @param src Source Image
 * @param means Discrete colors
 * @return Comicbook image
 */
static cv::Mat process_image(cv::Mat src, cv::Mat means)
{
    cv::Mat image(src, Range::all(), Range(120, 520));

    cv::Mat canny_overlay, halftone_overlay, posterized_image;
    pthread_t canny_t, halftone_t, posterized_t;

    // Create thread for each effect
    struct canny_thread_args canny_args = { &image, &canny_overlay };
    pthread_create(&canny_t, NULL, &canny_thread, &canny_args);

    struct halftone_thread_args halftone_args = { &image, &halftone_overlay };
    pthread_create(&halftone_t, NULL, &halftone_thread, &halftone_args);

    struct posterized_thread_args posterized_args = { &image, &means, &posterized_image };
    pthread_create(&posterized_t, NULL, &posterized_thread, &posterized_args);

    // Join threads
    pthread_join(canny_t, NULL);
    pthread_join(halftone_t, NULL);
    pthread_join(posterized_t, NULL);

    // Overlay effects before returning result
    Mat result = Effects::overlay(canny_overlay, halftone_overlay, posterized_image);
    return result;
}

/**
 * Thread for processing an image asynchronously
 * @param arg Pipeline* to parent object
 * @return NULL
 */
static void* pipeline_thread(void* arg)
{
    Pipeline* pipe = (Pipeline*)arg;

    struct Frame frame = pipe->capture->getFrame(pipe->last_frame);
    pipe->last_frame = frame.frame_num;
    cv::Mat result = process_image(frame.image, pipe->kmeans_src->getMeans());

    pthread_mutex_lock(&(pipe->mutex));
    pipe->result = result;
    pipe->running = false;
    pthread_cond_signal(&(pipe->cond));
    pthread_mutex_unlock(&(pipe->mutex));

    return NULL;
}

Pipeline::Pipeline(ImageCapture* capture, Kmeans* kmeans_src)
    : capture(capture)
    , kmeans_src(kmeans_src)
    , running(false)
    , last_frame(0)
{
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond, NULL);
}

Pipeline::~Pipeline() { join(); }

/**
 * Take latest image and means and process an image
 */
void Pipeline::start()
{
    pthread_mutex_lock(&mutex);
    if (!running) {
        running = true;
        pthread_create(&thread, NULL, &pipeline_thread, this);
    }
    pthread_mutex_unlock(&mutex);
}

/**
 * Wait for latest processing image
 * @return Processed image
 */
cv::Mat Pipeline::join()
{
    Mat ret;
    pthread_mutex_lock(&mutex);
    while (running) {
        pthread_cond_wait(&cond, &mutex);
    }
    pthread_join(thread, NULL);
    pthread_mutex_unlock(&mutex);
    return result;
}