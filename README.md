Lorenzo Giuliani, Roberto Minervini - Università Politecnica delle Marche, A.A. 2017/2018.

__Esame di Calcolatori Elettronici e Reti di Calcolatori.__


# 1. CV_cer_project

__CV_cer_project__ is a computer vision experiment.

It recognizes a target object of our choice and evaluates its distance from camera and its trajectory.
We are using the open-source [__*OpenCV*__](https://opencv.org) libraries which work at both _CPU_ level and _GPU_ level, by making use of [__*NVIDIA CUDA*__](https://developer.nvidia.com/cuda-toolkit) libraries.

The repository contains all source files but the utilized third-party libraries, and a short video demonstration of the software's functioning.

![Alt target](https://image.ibb.co/dwG0L0/small.png)
> Target Object

#### 1.1. Requirements

_OpenCV_ w/ _OpenGL_ + _CUDA_ support, _OpenGL_ + _CUDA_ capable _GPU_.

#### 1.2. Development Environment

The computer systems used during development and testing phases have the following specs, thus the experiment's results refer to these configurations:

|HW/SW|Setup #1|Setup #2|
|-|-|-|
|__CPU__|_Intel® Core™ i7-3632QM_ (_4 Cores_)|_Intel® Pentium® Dual-Core E5400_ (_2 Cores_)|
|__GPU__|_NVIDIA GeForce GT 630M_|_NVIDIA GeForce 310_|
|__RAM__|_6 GB DDR3_|_4 GB DDR3_|
|__OS__|_Windows 10 Home x64_|_Windows 10 Home x64_|
|__*VisualStudio* version__|_14 2015 x64_|_12 2013 x64_|
|__*OpenCV* version__|_3.2.0_|_3.1.0_|
|__*CUDA* toolkit__|_8.0.61_|_6.5.14_|

Not being recent systems to today's standards, especially the latter, there have been issues of compatibility and environment setups between the two. So, results may differ greatly on more recent hardware.


## 2. What's Behind It

Magari aggiungere qlcsina in più su object detection; qlcsina sulle istruzioni SIMD della Intel (quali MMX, AVX, SSE); forse qlcsina in più su SIMT.

_Object detection_ is a computer technology related to computer vision and image processing that deals with detecting instances of semantic objects of a certain class (such as humans, buildings, or cars) in digital images and videos. Every object class has its own special features that helps in classifying the class. Methods for object detection generally fall into either machine learning-based approaches or deep learning-based approaches.

_Machine Learning_ is a field of artificial intelligence which uses statistical techniques to give computer systems the ability to "learn" from data (for instance, progressively improve performance on a specific task), without being explicitly programmed to do so. On the other hand, _Deep Learning_ (also known as hierarchical learning) is about learning multiple levels of representations that correspond to different levels of abstraction, the levels forming a hierarchy of concepts that help to make sense of data such as images, sound, and text. Deep Learning can also be defined as a class of algorithms that use a cascade of multiple layers of nonlinear processing units for feature extraction and transformation, where each successive layer uses the output from the previous as input; Deep Learning can be of three different kinds: supervised, semi-supervised, unsupervised. 
_Supervised_ learning is based on learning a function that maps an input to an output based on example input-output pairs; it infers a function from completely labeled training data consisting of a set of training examples. _Unsupervised_ learning works through test data that has not been labeled, classified or categorized, by identifying commonalities in the data and by reacting based on the presence or absence of such commonalities in each new piece of data. _Semi-supervised_ learning falls in between, by making use of both labeled and unlabeled data for training (typically a small amount of labeled data with a large amount of unlabeled data).

_OpenCV_ libraries use a deep-learning approach and, to be more precise, a supervised learning method via given data samples. _OpenCV_'s `CascadeClassifier` class allows to detect objects in a video stream. Before the detection stage can happen, the `CascadeClassifier` itself needs to be trained with a, so called, boosted cascade of weak classifiers. In order to train this boosted cascade of weak classifiers, we need a set of positive samples (images containing actual objects we want to detect) and a set of negative samples (images containing everything we want not to detect). Negative samples are taken from arbitrary images, whereas positive samples can be created using the `opencv_createsamples` application.

In addition, _OpenCV_ libraries can be built to provide additional support for _OpenGL API_ and _NVIDIA CUDA_ libraries; such additions allow to utilize specific _GPU_ classes, to perform a number of operations on _GPU_ memory rather than on the system memory.

The _Graphics Processing Unit_ ([_GPU_](https://en.wikipedia.org/wiki/Graphics_processing_unit)) is a specialized electronic circuit designed to rapidly manipulate and alter memory to accelerate the creation of images in a frame buffer intended for output to a display device. Modern _GPUs_ are very efficient at manipulating computer graphics and image processing since their highly parallel structure makes them more efficient than general-purpose _CPUs_ for algorithms that process large blocks of data in parallel. In a personal computer, a _GPU_ can be present on a video card or embedded on the motherboard.

Speaking of _Parallel Computing_, _NVIDIA_ has worked on a different approach to the matter, by providing the _SIMT_ architecture. The _SIMT_ (_Single Instruction Multiple Threads_) architecture sits in between the _SIMD_ (_Single Instruction Multiple Data_) and the _SMT_ (_Simultaneous Multi Threading_) ones; in fact, it sort of takes into account both the functionalities of said architectures. On SIMD architectures, the same instruction is performed on a data array; on a SMT approach, many different threads are able to operate in parallel. So, SIMT is based on the principle that if a thread requires a data transfer, the _GPU_ selects another thread to execute; this due to the fact that, if there are so many threads to execute, it is highly unlikely that all threads are blocked on a data transfer. Thus it's all about keeping the _GPU_ always busy during memory transfers (this is called _latency hiding_). In the field of SIMT architectures, _NVIDIA_'s solution came in the form of a platform known as _CUDA_. This _Compute Unified Device Architecture_ is a programming model for parallel computing on _GPU_. _CUDA_ is only available for use on _NVIDIA_ proprietary cards, but there are also alternative platforms such as _OpenCL_, which provide similar solutions for different _GPUs_.


## 3. How We Made It

#### 3.1. - Cascade Generation

First of all, it is necessary to generate the `cascade.xml` file containing the comparison parameters, which is paramount in identifying one or more instances of the objects, present in the positives, inside each video frame. To this end, we have produced and collected a large number of positives, such as:

![Alt positive_img](https://image.ibb.co/ha6ovA/1.jpg)

and an even higher number of negatives not containing our object of interest, such as:

![Alt negative_img](https://image.ibb.co/i0GVaA/1-1.jpg)
> _Photograph taken among the university premises_

Next, we generated the _vector file_ of the positives using the `opencv_createsamples` application, with the following Command Prompt command and requested arguments:

`opencv_createsamples -vec C:\opencv\build\x64\vc14\bin\positive_samples.vec -w 41 -h 48 -info C:\opencv\build\x64\vc14\bin\info.dat -num 410`

The `-info` argument is to be filled with a correct path to a description file of marked up images collection, which we have manually prepared. Such description file contains a list of all available positives and selected regions of interest (specified by pixels) of our object instances
`-num` simply specifies the number of positives available for the creation of the _.vec_ file, while `w` and `h` respectively are the width and height of the cropped, gray-scale images that will reside in the .vec file.

With all training data being successfully prepared, we finally trained our boosted cascade of weak classifiers with the provided _OpenCV_ tool, in order to generate the fundamental `cascade.xml` file:

`opencv_traincascade -data C:\opencv\build\x64\vc14\bin\data\ -vec C:\opencv\build\x64\vc14\bin\positive_samples.vec -bg C:\opencv\build\x64\vc14\bin\info.txt
-numPos 350 -numNeg 690 -numStages 40 -precalcValBufSize 1024 -precalcIdxBufSize 1024 -numThreads 2 -stageType BOOST -featureType LBP
-w 41 -h 48 -bt GAB -minHitRate 0.995 -maxFalseAlarmRate 0.5 -weightTrimRate 0.95 -maxDepth 1 -acceptanceRatioBreakValue 0.0000412`

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

* no arbitrary dimensions support (only 2D)

* no functions that return references to their data (because references on _GPU_ are not valid for _CPU_)

* no expression templates technique support

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

At the end of the calibration phase, an output `_data/calibration_output.xml` file is generated. The core of our interest is `<camera_matrix>` (_Camera Matrix_), which describes the mapping of a pinhole camera from 3D points in the world to 2D points in an image. Here are represented `fx` and `fy`, respectively the focal length on the abscissas and on the ordinates (in pixels) of the camera in use. These values ​​are also linked by the equation `fy = fx * a`, where `a` is the _aspect ratio_, defined among the calibration parameters. If `a = 1` then `fy = fx = F`, where `F` is the _focal length_ (in pixels) of the camera.

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

The focal length is used to approximate the _real distance_ we want to calculate (`D`, in millimeters) of the object identified in the matrix, based on the perceived (`P`, in pixels) and the real (`W`, in millimeters) dimensions of the object itself, as defined in `#define W 75` in `main.cpp`:

`D = (W * F) / P`.

#### 3.4. - _CPU_

The part of code related to the functioning of the software on _CPU_ is contained entirely in the `process_cpu.cpp` file. This contains the `main_logic_cpu()` function which, having received an instance of our `cv::CascadeClassifier` initialized in the `main()` method as an input parameter, proceeds to the analysis of the captured frame, which is also another required input parameter of this method:

```javascript
Mat fgray;
cvtColor(frame, fgray, COLOR_BGR2GRAY);
equalizeHist(fgray, fgray);
```

`cvtColor()` converts the contents of a `Mat` instance to a specific property (in this case, `COLOR_BGR2GRAY` converts RGB images to grayscale) and stores the results inside the given `Mat fgray` output. The contents of `fgray` are once again processed by `equalizeHist()`, which equalizes the histogram of the contained grayscale images; this particular method is strictly necessary for our purposes, because it increases the global contrast of the images and enhances details that can be determining for the detection of our object of interest. As a matter of fact, "histogram equalization often produces unrealistic effects in photographs; however it is very useful for scientific images like thermal, satellite or x-ray images" (see [link](https://en.wikipedia.org/wiki/Histogram_equalization) for further details).

Next we define:

```javascript
std::vector<Rect> targets;
std::vector<Point> centers;
std::vector<double> distances;
```

these objects respectively are the vector designed to contain one or more instances of the target identified in the frame, the vector containing the center of those same objects and the one containing the distance of each target from camera. It follows that `dim(targets) = dim(centers) = dim(distances)`, because in each frame we have a center for every objective, with related distance.

```javascript
cascade.detectMultiScale(fgray, targets, 1.1, 5, ...);
std::vector<Point3d> pos;
```

The `detectMultiScale()` method, owned by the `cv::CascadeClassifier` element given in input, is responsible for the identification of one or more instances of the target object, inserted at the end of the vector `targets`. Following, the other parameters, of which the main ones are: _scale factor_, which specifies how much the image size is reduced at each image scale, it can resize a larger object towards a smaller one, making it detectable for the algorithm; _min neighbours_, which specifies how many neighbors each candidate rectangle should have to retain it. This is used to eliminate false positives.

In the event that only one instance of the object is detected in a frame, the 3D position of the object will be inserted into `pos`, an array that contains all the last 3D positions of the target, thus mapping its trajectory:

```javascript
size_t i = targets.size();
for (size_t j = 0; j < i; j++ ) {
	Point center(...);
	distances.push_back(...);
	centers.push_back(center);
	if(i == 1) { 	
		pos.push_back(Point3d(centers[j].x, centers[j].y, distances[j]));
	}
}
if (i > 1) {
	Point3d last_tr = Point3d(...);
	double dist = norm(last_tr - Point3d(centers[0].x, centers[0].y, distances[0]));
	size_t index = 0;
	for (size_t j = 1; j < i; j++) {
		if (norm(last_tr - Point3d(centers[j].x, centers[j].y, distances[j])) <= dist) {
			dist = norm(last_tr - Point3d(centers[j].x, centers[j].y, distances[j]));
			index = j;
		}
	}
	pos.push_back(Point3d(centers[index].x, centers[index].y, distances[index]));
}
```

for each target detected we calculate its real distance by simply applying the formula described at paragraph _3.2._ and we fill `distances`. If multiple instances are detected, we take the position of the object dimensionally closer to `Point3d last_tr = Point3d(pos[pos.size() - 1].x, pos[pos.size() - 1].y, pos[pos.size() - 1].z)` as the most likely position; in order to find it, we calculate the norm of the difference between `last_tr` and `Point3d(centers[j].x, centers[j].y, distances[j])`, with `j = 0, 1, ..., targets.size()`. To do so, we utilize the _OpenCV_ method `norm()`.

In `main_logic_cpu()` we later define, by an appropriate transformation, the position of the target in the plane of the spaces, setting the axis `z` at a distance of _500mm_, so we can render its coordinates on the frame.

#### 3.5. - _GPU_

On the _GPU_ side of things, matters are slightly different; inside `main_logic_gpu()` we make use of specific _GPU_ and/or _CUDA_ classes made available by _OpenCV_.
One of the main differences of `main_logic_gpu()` from its _CPU_ counterpart, is the use of `cv::cuda::CascadeClassifier` instead of the classic `cv::CascadeClassifier`, where the _CUDA_ one permits cascade classification on a _GPU_. However, we must point out that there is no clear advantage in using such class. Actually, it appears that the `cv::cuda::CascadeClassifier` class is no longer mantained and performs worse during detection, unlike `cv::CascadeClassifier` (more about at paragraph _4._).

In `main_logic_gpu()` we also introduce an instance of the `GpuMat` class, instead of `Mat`. More precisely, in the following code:

```javascript
GpuMat frame_gpu(frame);
GpuMat fgray;
```

the first line of code creates a `GpuMat` instance named `frame_gpu`, filled with the content of a `Mat frame` object which is passed as an argument of the `main_logic_gpu()` method; by doing so, we transfer our data from system memory to _GPU_ memory. 

In the second line of code we showed above, we create another instance of `GpuMat` named `fgray`; it is going to be utilized by the following methods, which manipulate the content of `GpuMat frame_gpu` and store the results inside `GpuMat fgray`:

```javascript
std::vector<Rect> targets;
...
cv::cuda::cvtColor(frame_gpu, fgray, COLOR_BGR2GRAY);
cv::cuda::equalizeHist(fgray, fgray);
cascade_gpu->setScaleFactor(1.1);
cascade_gpu->setMinNeighbors(5);
cascade_gpu->detectMultiScale(fgray, fgray);
cascade_gpu->convert(fgray, targets);
```

The first two methods simply are _GPU_ equivalents of the ones called inside `main_logic_cpu()`.

Now, we detail the differences of the `cv::cuda::CascadeClassifier` method `detectMultiScale()`; unlike the `cv::CascadeClassifier` one used in `main_logic_cpu()`, the _GPU_ version of `detectMultiScale()` requests just two arguments, the input matrix of frames and the output buffer where to store the detected objects (in the form of rectangles); in the case of `detectMultiScale(fgray, fgray)` we use `GpuMat fgray` as both the input matrix and the output buffer. The 'missing' parameters, the ones we have seen being added to `detectMultiScale()` in `main_logic_cpu()`, are actually defined before this method. We are talking about `setScaleFactor(1.1)` and `setMinNeighbors(5)`; their values exactly mirror those present inside `main_logic_cpu()` in `detectMultiScale()`. Changes to such values respectively affect the quality and quantity of detections, and the required minimum size for an object to be detected.

Finally, we convert the objects array from the internal representation (`GpuMat`) to a standard vector of rectangles (`std::vector<Rect> targets`).

The rest of the code mirrors that of `main_logic_cpu()`, although it is worth noting that all the on-screen strings we show on the frames can only be written on a `Mat` instance, not on a `GpuMat` object. To this end, we utilize the given `Mat frame` requested by our `main_logic_gpu()` method, by filling it with all the necessary strings for displaying our detection details. At the very end:

```javascript
frame_gpu.upload(frame);
return frame_gpu;
```

we upload all the contents of this loaded `Mat frame` object to the original `GpuMat frame_gpu`. Then, `main_logic_gpu()` returns an instance of `GpuMat` unlike `main_logic_cpu()`, which returns an instance of `Mat`. By design, this allowed us to have and use a `GpuMat` instance for the `imshow()` method called inside our `main()`.


## 4. Results

To distinctly calculate the execution time of the program for _CPU_ and _GPU_, we have used the _C_ library `time.h`:

```javascript
clock_t t0 = clock();
...
clock_t t1 = clock() - t0;
```

Variable `clock_t t1` indicates the time gap between this assignment and the initialization of` clock_t t0`.

Below we show a table with snapshots of the software in execution:

|Resource|Setup #1|Setup #2|
|-|-|-|
|__*CPU* Multi-Core__| _Average Execution Time_ ≅ 35 ms ![Alt setup1 multicpu](https://i.ibb.co/Wy6d28c/cpumulticore1.png) | _Average Execution Time_ ≅ 205 ms ![Alt setup2 multicpu](https://i.ibb.co/VCmB27d/cpumulticore2.png) |
|__*CPU* Single-Core__| _Average Execution Time_ ≅ 105 ms ![Alt setup1 singlecpu](https://i.ibb.co/5styNXK/cpusinglecore1.png) | _Average Execution Time_ ≅ 250 ms ![Alt setup2 singlecpu](https://i.ibb.co/dKc2B6V/cpusinglecore2.png) |
|__*GPU*__| _Average Execution Time_ ≅ 45 ms ![Alt setup1 gpu](https://i.ibb.co/7YNsBPR/gpu1.png) | _Average Execution Time_ ≅ 220 ms ![Alt setup2 gpu](https://i.ibb.co/CzxPbbd/gpu2.png) |

The _CPU Single-Core_ situation was achieved by setting the _CPU_ affinity of the process to one core only, via the _Task Manager_ of MS Windows.

_Setup #2_ (see paragraph _1.2._) does not show noticeable performance changes between all modes, due to its hardware. _Setup #1_ clearly shows a huge difference in performance, and it is a better example for our aim. We also want to point out that execution times on _CPU_ can vary due to the presence of processes running in background on the operating system. While hardware definetely affects the execution time, any additional softwares can take up important resources and slow down the entire execution on _CPU_.

As mentioned previously at paragraph _3.5._, there is no clear advantage in using a specialized `cv::cuda::CascadeClassifier` for _GPU_ use; in fact, as of _OpenCV_ version _3.1.0_, there have been reports that such _GPU_ implementation can be slower, due to bottlenecks when dealing with large resolution images. Moreover, having built _OpenCV_ libraries through _CMake_, support for processor supplementary capabilities (namely the _MMX_, _SSE_, _SSE2_, _SSE3_, _AVX_ and _AVX2_ SIMD instruction sets designed by Intel) has been included by default; this has implied a boost of performance on the _CPU_ side of things, when using multi-core systems (see [link](https://github.com/opencv/opencv/issues/6693) for further details).

To all effects we can conclude that a multi-core system, with all its _CPUs_, can perform better than a _GPU_, in our case.


__edited by__
__Lorenzo Giuliani, Roberto Minervini__
