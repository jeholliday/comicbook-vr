#include "pipeline.h"
#include "effects.h"

struct canny_thread_args {
    cv::Mat* src;
    cv::Mat* result;
};

static void* canny_thread(void* arg)
{
    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

    auto args = (struct canny_thread_args*)arg;
    *(args->result) = Effects::canny(*(args->src));

    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    std::cout << "Canny: " << (end - start).count() << " ms" << std::endl;

    return NULL;
}

struct halftone_thread_args {
    cv::Mat* src;
    cv::Mat* result;
};

static void* halftone_thread(void* arg)
{
    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

    auto args = (struct halftone_thread_args*)arg;
    *(args->result) = Effects::halftone(*(args->src));

    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    std::cout << "Halftone: " << (end - start).count() << " ms" << std::endl;

    return NULL;
}

struct posterized_thread_args {
    cv::Mat* src;
    cv::Mat* centers;
    cv::Mat* result;
};

static void* posterized_thread(void* arg)
{
    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

    auto args = (struct posterized_thread_args*)arg;
    *(args->result) = Effects::blur(Effects::posterize(*(args->src), *(args->centers)));

    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    std::cout << "Posterize: " << (end - start).count() << " ms" << std::endl;

    return NULL;
}

static cv::Mat process_image(cv::Mat src, cv::Mat means)
{
    cv::Mat image(src, Range::all(), Range(120, 520));

    cv::Mat canny_overlay, halftone_overlay, posterized_image;
    pthread_t canny_t, halftone_t, posterized_t;

    struct canny_thread_args canny_args = { &image, &canny_overlay };
    pthread_create(&canny_t, NULL, &canny_thread, &canny_args);

    // halftone_overlay = cv::Mat::zeros(image.size(), image.type());
    struct halftone_thread_args halftone_args = { &image, &halftone_overlay };
    pthread_create(&halftone_t, NULL, &halftone_thread, &halftone_args);

    struct posterized_thread_args posterized_args = { &image, &means, &posterized_image };
    pthread_create(&posterized_t, NULL, &posterized_thread, &posterized_args);

    pthread_join(canny_t, NULL);
    pthread_join(halftone_t, NULL);
    pthread_join(posterized_t, NULL);

    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    Mat result = Effects::overlay(canny_overlay, halftone_overlay, posterized_image);
    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    std::cout << "Overlay: " << (end - start).count() << " ms" << std::endl;
    return result;
}

static void* pipeline_thread(void* arg)
{
    Pipeline* pipe = (Pipeline*)arg;

    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    struct Frame frame = pipe->capture->getFrame(pipe->last_frame);
    pipe->last_frame = frame.frame_num;
    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    std::cout << "Get Frame: " << (end - start).count() << " ms" << std::endl;

    start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    cv::Mat result = process_image(frame.image, pipe->kmeans_src->getMeans());
    end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    std::cout << "Process Image: " << (end - start).count() << " ms" << std::endl;

    pthread_mutex_lock(&(pipe->mutex));
    pipe->result = result;
    pipe->running = false;
    pthread_cond_signal(&(pipe->cond));
    pthread_mutex_unlock(&(pipe->mutex));
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

void Pipeline::start()
{
    pthread_mutex_lock(&mutex);
    if (!running) {
        running = true;
        pthread_create(&thread, NULL, &pipeline_thread, this);
    }
    pthread_mutex_unlock(&mutex);
}

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