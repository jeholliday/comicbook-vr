# Comic Book VR
Realtime vision processing to turn a camera feed into a comic book

## How to Build
```
git clone https://github.com/jeholliday/comicbook-vr.git
cd comicbook-vr
mkdir build
cd build
cmake ..
make
```

## Executables
CMake will automatically detect if CUDA is available and create make targets for building GPU enabled versions. Executables suffixed with "-cpu" use the OpenCV CPU implementation of k-means.
- dual: VR headset version for using dual cameras and the image processing pipeline.
- live: Single camera mode for testing.
- offline: Process a single image from file for testing.
