# video-diff
Describe the types and degrees of transformations performed on one given video to produce the other.

This code made up the core of my thesis project, "YouTrace: A Smartphone System for Tracking Video Modifications".

* video_dissimilarity.py - Partial implementation
* videodiff.cpp - Full implementation
 * (I translated the code from Python to C++ in order to compile it natively on Android.)

To compile and run on the desktop, link with OpenCV libraries; for example:
```
g++ videodiff.cpp -g -Wall -pedantic -std=c++11 -Ofast `pkg-config --libs --cflags opencv` -pthread -o videodiff
```
  
Evaluation results will be posted here once the thesis has been published.
