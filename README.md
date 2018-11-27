Lorenzo Giuliani, Roberto Minervini - Università Politecnica delle Marche, A.A. 2017/2018.
__Esame di Calcolatori Elettronici e Reti di Calcolatori.__


# 1. CV_cer_project

__CV_cer_project__ is a computer vision experiment.
It recognizes a target object of our choice and evaluates its distance from camera and its trajectory.
We are using the open-source [__*OpenCV*__](https://opencv.org) libraries which work at both _CPU_ level and _GPU_ level, by making use of [__*NVIDIA CUDA*__](https://developer.nvidia.com/cuda-toolkit) libraries.
The repository contains all source files but the utilized third-party libraries, and a short video demonstration of its functioning.

> ![Alt target](https://image.ibb.co/dwG0L0/small.png)
> Target Object

#### 1.1. Requirements

_OpenCV_ w/ _OpenGL_ + _CUDA_ support, _OpenGL_ + _CUDA_ capable _GPU_.

#### 1.2. Development Environment

The computer systems used during development and testing phases have the following specs, thus the experiment's results refer to these configurations:

|HW/SW|Setup #1|Setup #2|
|-|-|-|
|__CPU__|_Intel® Core™ i7-3632QM_ (_8 Cores_)|_Intel® Pentium® Dual-Core E5400_ (_2 Cores_)|
|__GPU__|_NVIDIA GeForce GT 630M_|_NVIDIA GeForce 310_|
|__RAM__|_6 GB_|_4 GB_|
|__OS__|_Windows 10 Home x64_|_Windows 10 Home x64_|
|__*VisualStudio* version__|_14 2015 x64_|_12 2013 x64_|
|__*OpenCV* version__|_3.2.0_|_3.1.0_|
|__*CUDA* toolkit__|_8.0.61_|_6.5.14_|

Not being recent systems to today's standards, especially the latter, there have been issues of compatibility and environment setups between the two. So, results may differ greatly on more recent hardware.


## 2. What's Behind It

_Object detection_ is a computer technology related to computer vision and image processing that deals with detecting instances of semantic objects of a certain class (such as humans, buildings, or cars) in digital images and videos. Every object class has its own special features that helps in classifying the class. Methods for object detection generally fall into either machine learning-based approaches or deep learning-based approaches.
_Machine Learning_ is a field of artificial intelligence which uses statistical techniques to give computer systems the ability to "learn" from data (for instance, progressively improve performance on a specific task), without being explicitly programmed to do so. On the other hand, Deep Learning (also known as hierarchical learning) is about learning multiple levels of representations that correspond to different levels of abstraction, the levels forming a hierarchy of concepts that help to make sense of data such as images, sound, and text. 
_Deep Learning_ can also be defined as a class of algorithms that use a cascade of multiple layers of nonlinear processing units for feature extraction and transformation, where each successive layer uses the output from the previous as input; Deep Learning can be of three different kinds: supervised, semi-supervised, unsupervised. 
_Supervised_ learning is based on learning a function that maps an input to an output based on example input-output pairs; it infers a function from completely labeled training data consisting of a set of training examples. _Unsupervised_ learning works through test data that has not been labeled, classified or categorized, by identifying commonalities in the data and by reacting based on the presence or absence of such commonalities in each new piece of data. _Semi-supervised_ learning falls in between, by making use of both labeled and unlabeled data for training (typically a small amount of labeled data with a large amount of unlabeled data).

_OpenCV_ libraries use a deep-learning approach and, to be more precise, a supervised learning method via given data samples. _OpenCV_'s `CascadeClassifier` class allows to detect objects in a video stream. Before the detection stage can happen, the `CascadeClassifier` itself needs to be trained with a, so called, boosted cascade of weak classifiers. In order to train this boosted cascade of weak classifiers, we need a set of positive samples (images containing actual objects we want to detect) and a set of negative samples (images containing everything we want not to detect). Negative samples are taken from arbitrary images, whereas positive samples can be created using the `opencv_createsamples` application.

In addition, _OpenCV_ libraries can be built to provide additional support for _OpenGL API_ and _NVIDIA CUDA_ libraries; such additions allow to utilize specific _GPU_ classes, to perform a number of operations on _GPU_ memory rather than on the system memory.

The _Graphics Processing Unit_ ([_GPU_](https://en.wikipedia.org/wiki/Graphics_processing_unit)) is a specialized electronic circuit designed to rapidly manipulate and alter memory to accelerate the creation of images in a frame buffer intended for output to a display device. Modern _GPUs_ are very efficient at manipulating computer graphics and image processing since their highly parallel structure makes them more efficient than general-purpose _CPUs_ for algorithms that process large blocks of data in parallel. In a personal computer, a _GPU_ can be present on a video card or embedded on the motherboard.
Speaking of _Parallel Computing_, _NVIDIA_ has worked on a different approach to the matter, by providing the _SIMT_ architecture. The _SIMT_ (_Single Instruction Multiple Threads_) architecture sits in between the _SIMD_ (_Single Instruction Multiple Data_) and the _SMT_ (_Simultaneous Multi Threading_) ones; in fact, it sort of takes into account both the functionalities of said architectures. On SIMD architectures, the same instruction is performed on a data array; on a SMT approach, many different threads are able to operate in parallel. So, SIMT is based on the principle that if a thread requires a data transfer, the _GPU_ selects another thread to execute; this due to the fact that, if there are so many threads to execute, it is highly unlikely that all threads are blocked on a data transfer. Thus it's all about keeping the _GPU_ always busy during memory transfers (this is called _latency hiding_). In the field of SIMT architectures, _NVIDIA_'s solution came in the form of a platform known as _CUDA_. This _Compute Unified Device Architecture_ is a programming model for parallel computing on _GPU_. _CUDA_ is only available for use on _NVIDIA_ proprietary cards, but there are also alternative platforms such as _OpenCL_, which provide similar solutions for different _GPUs_.


## 3. How We Made It

#### 3.1. - Cascade Generation

First of all, it is necessary to generate the `cascade.xml` file containing the comparison parameters, which is paramount in identifying one or more instances of the objects, present in the positives, inside each video frame. To this end, we have produced and collected a large number of positives, such as:

> ![Alt positive_img](https://image.ibb.co/ha6ovA/1.jpg)

and an even higher number of negatives not containing our object of interest, such as:

> ![Alt negative_img](https://image.ibb.co/i0GVaA/1-1.jpg)
> _Photograph taken among the university premises_

Next, we generated the _vector file_ of the positives using the `opencv_createsamples` application, with the following Command Prompt command and requested arguments:

`opencv_createsamples -vec C:\opencv\build\x64\vc14\bin\positive_samples.vec -w 41 -h 48 -info C:\opencv\build\x64\vc14\bin\info.dat -num 410`

The `-info` argument is to be filled with a correct path to a description file of marked up images collection, which we have manually prepared. Such description file contains a list of all available positives and selected regions of interest (specified by pixels) of our object instances
`-num` simply specifies the number of positives available for the creation of the _.vec_ file, while `w` and `h` respectively are the width and height of the cropped, gray-scale images that will reside in the .vec file.

With all training data being successfully prepared, we finally trained our boosted cascade of weak classifiers with the provided _OpenCV_ tool, in order to generate the fundamental `cascade.xml` file:

`opencv_traincascade -data C:\opencv\build\x64\vc14\bin\data\ -vec C:\opencv\build\x64\vc14\bin\positive_samples.vec -bg C:\opencv\build\x64\vc14\bin\info.txt -numPos 350 -numNeg 690 -numStages 40 -precalcValBufSize 1024 -precalcIdxBufSize 1024 -numThreads 2 -stageType BOOST -featureType LBP -w 41 -h 48 -bt GAB -minHitRate 0.995 -maxFalseAlarmRate 0.5 -weightTrimRate 0.95 -maxDepth 1 -acceptanceRatioBreakValue 0.0000412`

In the following link there is a rather complete explanation of the `opencv_traincascade` arguments, as provided by the [official _OpenCV_ documentation](https://docs.opencv.org/3.2.0/dc/d88/tutorial_traincascade.html).
We kept most of the default values with little variation; however, it's worth mentioning that we chose `-featureType LBP` instead of the default `HAAR` value. This due to the fact that an _LBP_ (_Local Binary Patterns_) cascade can be trained within minutes while, out of the box, the _HAAR_ cascade is about three times slower and, depending on the provided training data, it can take days to develop on mid-end systems. LBP is faster, but it features yield integer precision in contrast to HAAR, yielding floating point precision. However, quality of the cascade is highly variable as it mainly depends on the used training data and the selected training parameters. It's confirmed that it is possible to train an LBP-based classifier that will provide almost the same quality as a HAAR-based one. To our ends, LBP is a better solution, but we want to stress out that it can be difficult to obtain a working cascade if the provided positives and negatives are low in number and/or not so consistent in quality.

#### 3.2. - _MAIN_

After having generated a working `cascade.xml`, we took care of writing the source code, contained in its entirety in this repository. The core file is `main.cpp`, containing the `main()` method, responsible for the initialization of the software itself and the resources it uses, including the aforementioned `cascade.xml`. This `main()` is intended to generate a calibration window for the camera (further explained at paragraph _3.3._) and, subsequently, two windows with different properties that show the functioning of the software relative to use on _CPU_ and _GPU_.

To notice that we have created two instances for the following two classes:
```javascript
static cv::CascadeClassifier target_cascade;
static cv::Ptr<cv::cuda::CascadeClassifier> target_cascade_gpu;
```
The `CascadeClassifier` class is meant to detect objects in a video stream. The _CUDA_ version of that class simply performs such detection by exploiting _GPU_ memory rather than system memory (more details at paragraph _3.5._).
In order to load a .xml classifier file, we had to resort to command line parsing, so that we could specify a working path to the requested file (the `load()` method of `cv::CascadeClassifier` alone didn't work as intended). Here below an example of what can be found in the `main()`:
```javascript
fs::path p = "_data\\cascade.xml";
CommandLineParser parser(argc, argv,
	"{help h||}"
	"{target_cascade|" + fs::system_complete(p).string() + "|}");
String target_cascade_name = parser.get<string>("target_cascade");
if (!target_cascade.load(target_cascade_name)) {
	cerr << "ERROR: Can't load target cascade." << endl;
	return 1;
}
```

Back to the core of our program, the following objects are initialized once the calibration phase (explained at paragraph _3.3._) has been completed:
```javascript
Mat mainframe;
GpuMat mainframe_gpu;
VideoCapture capture(0);
```
`cv::Mat` is an object provided by _OpenCV_'s libraries and it represents an _n_-dimensional dense numerical single-channel or multi-channel array. We use it, introducing it with the name `Mat mainframe`, as the main container of the frames that we are going to acquire through our camera.
`cv::cuda::GpuMat mainframe_gpu` allows us to display our video frames via _GPU_ memory, thus not utilizing the _CPU_. However, we had to construct a window with the `WINDOW_OPENGL` property; in fact, only OpenGL windows allow for the visualization of `GpuMat` instances. Other types of window provided by _OpenCV_ (such as `WINDOW_NORMAL` and `WINDOW_AUTOSIZE`) exclusively utilize _CPU_ memory and act as placeholders for `Mat` instances only.
The `cv::cuda::GpuMat` interface matches the `cv::Mat` interface with the following limitations:
-no arbitrary dimensions support (only 2D)
-no functions that return references to their data (because references on _GPU_ are not valid for _CPU_)
-no expression templates technique support
However, such limitations do not hamper the real objective of this software.
Eventually we define an instance of `cv::VideoCapture`, an interface that takes care of communication with the _driver_ of a video device connected to the computer (in our case the `0` device, that is the _default_ one), which with the `>>` operator allows us to acquire frames from the device itself:
```javascript
while(true) {
	capture >> mainframe;
	...
	if(...) {
		mainframe = main_logic_cpu(mainframe, ...);
		imshow(mainframe);
	}
	else {
		mainframe_gpu = main_logic_gpu(mainframe, ...);
		imshow(mainframe_gpu);
	}
}
return 0;
```
this cycle is the heart of the software; here frames are repeatedly acquired by the camera and are appropriately processed according to whether the user wants to exploit the capacity of his own _CPU_ or _GPU_. This choice can be made by repeatedly pressing the key "_spacebar_", thus alternating between the two modes. 
Note that `mainframe_gpu` is defined by passing the acquired element of type `Mat` to `main_logic_gpu()`. The contents of this `Mat` element will be uploaded to _GPU_ memory within the mentioned method (more in detail at paragraph _3.5._). Finally, it is important to specify that, by pressing the "_ESC_" key, it is possible to exit the `while()` loop, thus returning `int 0`.
  
#### 3.3. - Calibration

In order to get the best measurement results, in particular the distance of the target object, a calibration of the camera we use must be carried out as accurately as possible. As it is not the real objective of our project, for this purpose we have adapted the following [source code](https://github.com/opencv/opencv/blob/3.4/samples/cpp/tutorial_code/calib3d/camera_calibration/camera_calibration.cpp) as given by _OpenCV_. Due to compatibility issues, bits of an older version of the linked source code have been borrowed for use on the older _OpenCV_ version 3.1.0, which has been utilized on one of the two Development Environments (see paragraph _1.2._ for further details).
The calibration parameters are defined in the `_data/calibration_input.xml` file. _OpenCV_ currently supports 3 different types of patterns; in our case a predetermined chessboard pattern is defined inside `_data/calibration_input.xml`. We provide a chessboard pattern in `PDF` at the following path: `extras/CHESSBOARD_pattern.pdf`; or it can also be downloaded directly from the [official source](https://docs.opencv.org/3.4/pattern.png).
It's recommended to use as many samples (frames, node `<Calibrate_NrOfFrameToUse>`) as possible, at least 20. The chessboard must be rigid, because if it's wrinkled or flopping around it will affect calibration. Moreover, the chessboard must take up as much of the camera view as possible. During calibration, the chessboard must be rotated and skewed.

At the end of the calibration phase, an output `_data/calibration_output.xml` file is generated. The core of our interest is `<camera_matrix>` (_Camera Matrix_), which describes the mapping of a pinhole camera from 3D points in the world to 2D points in an image. Here are represented `fx` e `fy`, respectively the focal length on the abscissas and on the ordinates (in pixels) of the camera in use. These values ​​are also linked by the equation `fy = fx * a`, where `a` is the _aspect ratio_, defined among the calibration parameters. If `a = 1` then `fy = fx = F`, where `F` is the _focal length_ (in pixels) of the camera.
Focal length's value automatically returns back to the `main()` method:
```javascript
return cameraMatrix.at<double>(0, 0) * s.aspectRatio;
```
If the user decides to ignore the calibration phase, then the `get_focal_length()` method is invoked, which reads the `calibration_output.xml` file, if present:
```javascript
FileStorage fs(...);
if (!fs.isOpened()) {
	return 560;
}
Mat cameraMatrix;
fs["camera_matrix"] >> cameraMatrix;
return cameraMatrix.at<double>(1, 1);
```
If an error occurs during reading (file does not exist, file is corrupt), a value of _default_ is returned; otherwise, unlike the previous `return`, the `fy` of the matrix camera is returned directly, since we have `fy = F`, again from the previous relation.
The focal length is used to approximate the _real distance_ we want to calculate (`D`, in `mm`) of the object identified in the matrix, based on the perceived (`P`, in `px`) and the real (`W`, in `mm`) dimensions of the object itself, as defined in `#define W 75` in `main.cpp`:
`D = (W * F) / P`.


__edited by__
__Lorenzo Giuliani, Roberto Minervini__
