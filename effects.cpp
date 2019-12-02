#include "effects.h"

Mat Effects::canny(Mat src)
{
  Mat src_gray;
  // Convert image to gray and blur it
  cvtColor( src, src_gray, COLOR_BGR2GRAY );
  cv::blur( src_gray, src_gray, Size(3,3) );

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
}

Mat Effects::overlay(Mat canny_overlay, Mat posterized_image, Mat halftone_overlay)
{
  // Combine Canny and Halftone
  Mat alpha = canny_overlay;
  Mat back = posterized_image;
  Mat fore;
  bitwise_not(canny_overlay, fore);
  addWeighted(fore, 0.5, halftone_overlay, 0.5, 0, fore);

  // Convert Mat to float data type
  fore.convertTo(fore, CV_32FC3);
  back.convertTo(back, CV_32FC3);

  // Normalize the alpha mask to keep intensity between 0 and 1
  alpha.convertTo(alpha, CV_32FC3, 1.0/255); //

  // Storage for output image
  Mat out = Mat::zeros(fore.size(), fore.type());

  // Multiply the fore with the alpha matte
  multiply(alpha, fore, fore);

  // Multiply the back with ( 1 - alpha )
  multiply(Scalar::all(1.0)-alpha, back, back);

  // Add the masked fore and back.
  add(fore, back, out);

  out /= brightness;
  out.convertTo(out, CV_8UC3, 255.0);
  return out;
}

Mat Effects::blur(Mat src)
{
  Mat new_image(src);
  //Apply Median Blur
  for ( int i = 1; i < 7; i = i + 2 )
    medianBlur (new_image, new_image, i);

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
    Mat new_image(src.rows, src.cols, src.type());

    pthread_t threads[NUM_THREADS];
    struct halftone_args args[NUM_THREADS];
    for (size_t i = 0; i < NUM_THREADS; ++i) {
        args[i].start_index = i * src.rows / NUM_THREADS;
        args[i].end_index = (i+1)*src.rows / NUM_THREADS;
        args[i].img = &src;
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
    auto args = (struct halftone_args*)arg;
    int nbhdSize = 9;

    int j;
    for (int i = args->start_index; i < args->end_index - nbhdSize; i++){
        j = 0;
        while (j < args->img->cols - nbhdSize){
            double nbhdSum = 0;
            Scalar nbhdSumColor = 0;
            for (int k = 0; k < nbhdSize; k++){
                Mat nbhdRow = args->gray_img->row(i+k).colRange(j, j+nbhdSize).clone();
                Mat nbhdRowColor = args->img->row(i+k).colRange(j, j+nbhdSize).clone();
                nbhdSum += sum(nbhdRow)[0];
                nbhdSumColor += sum(nbhdRowColor);
            }

            // Average
            double average = nbhdSum / nbhdSize;
            Scalar color_avg = nbhdSumColor / (int(nbhdSize));

            // Scale average into a circle radius intensity
            double max = (2.5/3) * 0.5 * nbhdSize;
            double scaled_intensity = max - (average / (255.0 * nbhdSize)) * max;

            // Draw a circle on the image
            circle(*(args->new_image), Point(j+nbhdSize/2.0, i+nbhdSize/2.0), scaled_intensity,
                       args->img->at<Vec3b>(i+nbhdSize/2.0, j+nbhdSize/2.0),-1);

            j += nbhdSize;
        }
        i += nbhdSize;
    }
    return nullptr;
}
