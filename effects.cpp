#include "effects.h"
#include "timing.h"

/**
 * Perform canny edge detection
 * @param src Source Image
 * @return Mat with detected edges
 */
Mat Effects::canny(Mat src)
{
    START_TIMING();
    Mat src_gray, detected_edges;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    cv::blur(src_gray, detected_edges, Size(3, 3));
    Canny(detected_edges, detected_edges, 30, 60, 3);

    STOP_TIMING("Canny");
    return detected_edges;
}

/**
 * Helper method for blurring an image
 * @param src Source Image
 * @return Blurred Image
 */
Mat Effects::blur(Mat src)
{
    Mat new_image(src);
    GaussianBlur(src, new_image, Size(3, 3), 0, 0);
    return new_image;
}

/**
 * Overlay Canny edges and halftone dots onto a posterized image
 * @return Combined image
 */
Mat Effects::overlay(Mat canny_overlay, Mat halftone_overlay, Mat posterized_image)
{
    START_TIMING();
    Mat fore;
    Mat halftone_mask;
    cvtColor(halftone_overlay, halftone_mask, COLOR_BGR2GRAY);

    cv::threshold(halftone_mask, halftone_mask, 1, 255, cv::THRESH_BINARY);
    add(canny_overlay, halftone_mask, fore);
    bitwise_not(fore, fore);

    Mat out;
    posterized_image.copyTo(out, fore);
    add(halftone_overlay, out, out);

    STOP_TIMING("Overlay");
    return out;
}

/**
 * Color an image using a reduced color set
 * @param src Source image
 * @param centers Discrete colors
 * @return Posterized image
 */
Mat Effects::posterize(Mat src, Mat centers)
{
    START_TIMING();
    Mat img;
    src.convertTo(img, CV_32FC3);
    img = img.reshape(3, img.total());
    centers = centers.t();
    centers = centers.reshape(3, centers.rows);

    Mat new_image(src.total(), 1, src.type());

    pthread_t threads[NUM_THREADS];
    struct posterize_args args[NUM_THREADS];
    for (size_t i = 0; i < NUM_THREADS; ++i) {
        args[i].start_index = i * src.total() / NUM_THREADS;
        args[i].end_index = min(src.total(), (i + 1) * src.total() / NUM_THREADS);
        args[i].img = &img;
        args[i].centers = &centers;
        args[i].new_image = &new_image;
        pthread_create(&threads[i], NULL, &posterize_thread, &args[i]);
    }
    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }
    new_image = new_image.reshape(3, src.rows);
    new_image = Effects::blur(new_image);
    STOP_TIMING("Posterize");
    return new_image;
}

/**
 * Thread to posterize a subset of an image
 * @param arg posterize_args*
 * @return NULL
 */
void* Effects::posterize_thread(void* arg)
{
    auto args = (struct posterize_args*)arg;
    for (int i = args->start_index; i < args->end_index; ++i) {
        Vec3f value = args->img->at<Vec3f>(i, 0);
        float best_distance = FLT_MAX;
        int32_t best_cluster = 0;
        for (int32_t cluster = 0; cluster < args->centers->rows; ++cluster) {
            Vec3f center = args->centers->at<Vec3f>(cluster, 0);
            const float distance = (value[0] - center[0]) * (value[0] - center[0]) + (value[1] - center[1]) * (value[1] - center[1])
                + (value[1] - center[1]) * (value[1] - center[1]);
            if (distance < best_distance) {
                best_distance = distance;
                best_cluster = cluster;
            }
        }

        args->new_image->at<Vec3b>(i, 0)[0] = args->centers->at<Vec3f>(best_cluster, 0)[0];
        args->new_image->at<Vec3b>(i, 0)[1] = args->centers->at<Vec3f>(best_cluster, 0)[1];
        args->new_image->at<Vec3b>(i, 0)[2] = args->centers->at<Vec3f>(best_cluster, 0)[2];
    }
    return nullptr;
}

/**
 * Create halftone effect from an image
 * @param src Source image
 * @return Image with halftone dots
 */
Mat Effects::halftone(Mat src)
{
    START_TIMING();
    Mat gray_src;
    cvtColor(src, gray_src, COLOR_BGR2GRAY);
    Mat new_image = cv::Mat::zeros(src.size(), src.type());

    pthread_t threads[NUM_THREADS];
    struct halftone_args args[NUM_THREADS];
    for (size_t i = 0; i < NUM_THREADS; ++i) {
        args[i].start_index = i * src.rows / NBHD_SIZE / NUM_THREADS;
        args[i].end_index = (i + 1) * src.rows / NBHD_SIZE / NUM_THREADS;
        args[i].src = &src;
        args[i].gray_img = &gray_src;
        args[i].new_image = &new_image;
        pthread_create(&threads[i], NULL, &halftone_thread, &args[i]);
    }
    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }
    STOP_TIMING("Halftone");
    return new_image;
}

/**
 * Thread to perform halftone on a subset of an image
 * @param arg halftone_args*
 * @return NULL
 */
void* Effects::halftone_thread(void* arg)
{
    auto args = (struct halftone_args*)arg;

    for (int i = NBHD_SIZE * args->start_index; i < NBHD_SIZE * args->end_index; i += NBHD_SIZE) {
        for (int j = 0; j + NBHD_SIZE < args->src->cols; j += NBHD_SIZE) {
            double nbhdSum = 0;
            for (int k = 0; k < NBHD_SIZE; k++) {
                Mat nbhdRow = args->gray_img->row(i + k).colRange(j, j + NBHD_SIZE).clone();
                nbhdSum += sum(nbhdRow)[0];
            }

            // Average
            double average = nbhdSum / NBHD_SIZE;

            // Scale average into a circle radius intensity
            double max = (2.0 / 3) * 0.5 * NBHD_SIZE;
            double scaled_intensity = max - (average / (255.0 * NBHD_SIZE)) * max;

            // Draw a circle on the image
            circle(*(args->new_image), Point(j + NBHD_SIZE / 2, i + NBHD_SIZE / 2), scaled_intensity,
                args->src->at<Vec3b>(i + NBHD_SIZE / 2, j + NBHD_SIZE / 2), -1);
        }
    }
    return nullptr;
}
