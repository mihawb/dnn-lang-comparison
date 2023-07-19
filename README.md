# dnn-lang-comparison
Language performance comparison in image classification with deep neural networks (fully connected, convolutional, MobileNet, ConvNeXt, among other models) with GPU utilization 

## Motivation
C++ is obviously and undeniably faster than Python when run on CPU. How do they compare, however, when all the heavy lifting is shifted to GPU? Not to mention there are multiple frameworks for deep neural networks - which one is the fastest, and which one is most tiresome to use? ...Does anyone still use Matlab?  
Those, among others, are the questions that were keeping me awake at night, so I started this project in hopes of answering them once and for all.

### No motivation anymore
This project was supposed to be my BSc thesis, but I guess I lost steam on it, especially while trying to get C++ to work (Rustacean here). Jokes aside, it just felt too repetitive, even more so that results aren't really anything to write ~~home~~ a thesis about. Now it just stands as proof that I know how to install nVidia drivers on Linux (and also that I know PyTorch and TensorFlow but the former is way more impressive, isn't it).

## Technologies used
All benchmarks were run on GTX 1050 with CUDA 11.8 with drives suitable for Ubuntu  
* Python (3.11)
  * PyTorch
  * TensorFlow
  * Numpy
  * Pandas
  * multiprocessing
* C++ (14)
  * cuDNN
  * CUDA
* Matlab (R2023a)

## Results
Refer to chapters 5 and 6 in the [final report](https://github.com/mihawb/dnn-lang-comparison/blob/main/analysis/final_report.pdf).  
Attaching some plots below as they draw attention.  
![training](https://github.com/mihawb/dnn-lang-comparison/assets/46073943/3623ba86-784f-4a83-b725-2b57db24b2d0)
![inference](https://github.com/mihawb/dnn-lang-comparison/assets/46073943/7265f291-598e-4cbc-9734-69b78ef1d71d)
**TL;DR**: PyTorch turned out to be not much slower than TensorFlow (sometimes even faster), but was way nicer to work with - it's very pythonic. I can't recommend it enough.
