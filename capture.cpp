#include "capture.h"

void* capture_thread(void* arg) {
  ImageCapture* capture = (ImageCapture*) arg;

  while(!capture->stopped){
    cv::Mat image;
    capture->cap >> image;
    pthread_mutex_lock(&(capture->mutex));
    image.copyTo(capture->image);
    capture->frame_num += 1;
    pthread_cond_broadcast(&(capture->cond));
    pthread_mutex_unlock(&(capture->mutex));
  }
  return NULL;
}

ImageCapture::ImageCapture(int id): cap(id), frame_num(0), stopped(false){
  pthread_mutex_init(&mutex, NULL);
  pthread_cond_init(&cond, NULL);
  pthread_create(&thread, NULL, &capture_thread, this);
}

ImageCapture::~ImageCapture(){
  stop();
}

struct Frame ImageCapture::getFrame(size_t last_frame){
  pthread_mutex_lock(&mutex);
  while(last_frame == frame_num){
    pthread_cond_wait(&cond, &mutex);
  }
  cv::Mat latest;
  image.copyTo(latest);
  size_t frame = frame_num;
  pthread_mutex_unlock(&mutex);
  return {latest, frame};
}

void ImageCapture::stop(){
  pthread_mutex_lock(&mutex);
  if(!stopped){
    stopped = true;
    pthread_mutex_unlock(&mutex);
    pthread_join(thread, NULL);
    cap.release();
  }else{
    pthread_mutex_unlock(&mutex);
  }
}