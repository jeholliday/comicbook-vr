#include "capture.h"

void* capture_thread(void* arg) {
  ImageCapture* capture = (ImageCapture*) arg;

  while(!capture->stopped){
    //std::cout << "Reported FPS: " << capture->cap.get(cv::CAP_PROP_FPS) << std::endl;
    cv::Mat image;

    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    capture->cap >> image;
    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    //td::cout << "Get Image: " << (end-start).count() << " ms" << std::endl;

    start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    cv::resize(image, image, cv::Size(640, 480));
    cv::flip(image, image, -1);
    end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    //std::cout << "Resize/Flip: " << (end-start).count() << " ms" << std::endl;

    pthread_mutex_lock(&(capture->mutex));
    //std::cout << "Acquired new image" << std::endl;
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

  cap.set(cv::CAP_PROP_FPS, 30);
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
    pthread_mutex_lock(&mutex);
    cap.release();
    frame_num+=1;
    pthread_cond_broadcast(&cond); // Wakeup anybody waiting
  }
  pthread_mutex_unlock(&mutex);
}