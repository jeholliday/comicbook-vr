#include "kmeans.h"

void* kmeans_thread(void* arg){
  Kmeans* parent = (Kmeans*) arg;

  size_t last_frame = 0;
  while(!parent->stopped){
    struct Frame frame = parent->src->getFrame(last_frame);
    last_frame = frame.frame_num;
    cv::Mat new_means = kmeans(frame.image, parent->means, parent->k, parent->num_iterations);

    pthread_mutex_lock(&(parent->mutex));
    new_means.copyTo(parent->means);
    pthread_cond_broadcast(&(parent->cond));
    pthread_mutex_unlock(&(parent->mutex));
  }
}

Kmeans::Kmeans(int k, int num_iterations, ImageCapture* src): k(k), num_iterations(num_iterations), src(src), stopped(false){
  pthread_mutex_init(&mutex, NULL);
  pthread_cond_init(&cond, NULL);
  pthread_create(&thread, NULL, &kmeans_thread, this);
}

Kmeans::~Kmeans(){
  stop();
}


cv::Mat Kmeans::getMeans(){
  pthread_mutex_lock(&mutex);
  while(means.empty()){
    pthread_cond_wait(&cond, &mutex);
  }
  cv::Mat latest;
  means.copyTo(latest);
  pthread_mutex_unlock(&mutex);
  return latest;
}

void Kmeans::stop(){
  pthread_mutex_lock(&mutex);
  if(!stopped){
    stopped = true;
    pthread_mutex_unlock(&mutex);
    pthread_join(thread, NULL);
  }else{
    pthread_mutex_unlock(&mutex);
  }
}
