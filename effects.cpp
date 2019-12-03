#include "effects.h"

Mat Effects::canny(Mat src)
{
  /*
  Mat src_gray;
  // Convert image to gray and blur it
  cvtColor( src, src_gray, COLOR_BGR2GRAY );
  //cv::blur( src_gray, src_gray, Size(3,3) );

  Mat canny_output;
  std::vector<std::vector<Point> > contours;
  std::vector<Vec4i> hierarchy;

  // Detect edges using canny
  Canny( src_gray, canny_output, 30, 60, 3 );
  // Find contours
  findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

  // Draw contours
  Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
  for(size_t i = 0; i< contours.size(); i++ )
  {
    Scalar color = Scalar(255,255,255);
    drawContours( drawing, contours, i, color, 1, 8, hierarchy, 0, Point() );
  }

  return drawing;
   */

  Mat src_gray, detected_edges;

  cvtColor( src, src_gray, COLOR_BGR2GRAY );

  cv::blur(src_gray, detected_edges, Size(3,3));
  Canny( detected_edges, detected_edges, 30, 60, 3);
  return detected_edges;
}

Mat Effects::overlay(Mat canny_overlay, Mat halftone_overlay, Mat posterized_image)
{
  Mat fore;
  Mat halftone_mask;
  cvtColor(halftone_overlay, halftone_mask, COLOR_BGR2GRAY);
  //cvtColor(canny_overlay, canny_overlay, COLOR_BGR2GRAY);

  cv::threshold(halftone_mask, halftone_mask, 1, 255, cv::THRESH_BINARY);
  add(canny_overlay, halftone_mask, fore);
  bitwise_not(fore, fore);

  Mat out; //= Mat::zeros(fore.size(), fore.type());

  posterized_image.copyTo(out, fore);

  add(halftone_overlay, out, out);

  return out;
}

Mat Effects::blur(Mat src)
{
  Mat new_image(src);
  /*//Apply Median Blur
  for ( int i = 1; i < 7; i = i + 2 )
    medianBlur (new_image, new_image, i);
  */
  GaussianBlur(src, new_image, Size(3,3), 0, 0);

  return new_image;
}

Mat Effects::posterize(Mat src, Mat centers)
{
  Mat img;
  src.convertTo(img, CV_32FC3);
  img = img.reshape(3, img.total());
  centers = centers.t();
  centers = centers.reshape(3,centers.rows);

  Mat new_image(src.total(), 1, src.type());

  pthread_t threads[NUM_THREADS];
  struct posterize_args args[NUM_THREADS];
  for(size_t  i = 0; i < NUM_THREADS; ++i){
    args[i].start_index = i * src.total() / NUM_THREADS;
    args[i].end_index = min(src.total(), (i+1) * src.total() / NUM_THREADS);
    args[i].img = &img;
    args[i].centers = &centers;
    args[i].new_image = &new_image;
    pthread_create(&threads[i], NULL, &posterize_thread, &args[i]);
  }
  for(int i = 0;i < NUM_THREADS; ++i){
    pthread_join(threads[i], NULL);
  }
  new_image = new_image.reshape(3, src.rows);
  return new_image;
}

void* Effects::posterize_thread(void* arg) {
  auto args = (struct posterize_args*)arg;
  for(int i=args->start_index;i<args->end_index;++i){
    Vec3f value = args->img->at<Vec3f>(i,0);
    float best_distance = FLT_MAX;
    int32_t best_cluster = 0;
    for (int32_t cluster = 0; cluster < args->centers->rows; ++cluster) {
      Vec3f center = args->centers->at<Vec3f>(cluster, 0);
      const float distance = (value[0]-center[0])*(value[0]-center[0]) + (value[1]-center[1])*(value[1]-center[1]) + (value[1]-center[1])*(value[1]-center[1]);
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


Mat Effects::halftone(Mat src) {
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
    return new_image;
}

void* Effects::halftone_thread(void* arg) {
    auto args = (struct halftone_args*) arg;

    for (int i = NBHD_SIZE * args->start_index; i < NBHD_SIZE * args->end_index; i+= NBHD_SIZE){
        for (int j = 0; j + NBHD_SIZE < args->src->cols; j+= NBHD_SIZE){
            double nbhdSum = 0;
            for (int k = 0; k < NBHD_SIZE; k++){
                Mat nbhdRow = args->gray_img->row(i+k).colRange(j, j+NBHD_SIZE).clone();
                nbhdSum += sum(nbhdRow)[0];
            }

            // Average
            double average = nbhdSum / NBHD_SIZE;

            // Scale average into a circle radius intensity
            double max = (2.0/3) * 0.5 * NBHD_SIZE;
            double scaled_intensity = max - (average / (255.0 * NBHD_SIZE)) * max;

            // Draw a circle on the image
            circle(*(args->new_image), Point(j+NBHD_SIZE/2, i+NBHD_SIZE/2), scaled_intensity,
                       args->src->at<Vec3b>(i+NBHD_SIZE/2, j+NBHD_SIZE/2),-1);
        }
    }
    return nullptr;
}
