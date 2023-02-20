# What the hell is this?

![reaction-diffusion.png](reaction-diffusion.png)

# Why it exists?

* To try ChatGPT (first commit, AVX code)
* It looks cool :-)
* Potential playground for trying some libraries (TaskFlow)
* Playground for trying AVX instuction set
* Playground for high performance computing / image processing and parallelism

Conversation with ChatGPT (or major parts) are in [chatgpt-reaction-diffusion.md](chatgpt-reaction-diffusion.md) file. It helped a lot with initial code (except some constant were wrong, it used one array for input and output and it took like 2 hours to fix, cause wrong choice of constants was not obvious). Generated simplified AVX code was super useful.

Problem is that ChatGPT gives answers that seem right and belieavable, but sometimes they are oversimplified, made up or just wrong. If you speak with human, you can somehow feel uncertainity about answers or hiding his lack of knowledge or competence in area. ChatGPT just gives wrong answer and writes some extra facts on top of it.

# Links

* https://mrob.com/pub/comp/xmorphia/index.html (likely original idea)
* https://www.karlsims.com/rd.html (explantion)
* https://github.com/topics/reaction-diffusion
* https://jasonwebb.github.io/reaction-diffusion-playground/app.html (one of online simulators, initial constants from here, good to see number of iterations)

# Requirements

## Knowledge

* CMake basics
* C++ basics

## Third party libraries

* OneTBB (Intel's OneAPI Thread Building Blocks) https://github.com/oneapi-src/oneTBB/releases
* OpenCV https://opencv.org/releases/
* FmtLib https://github.com/fmtlib/fmt (since v0.2)
* TaskFlow 3.4.0 https://github.com/taskflow/taskflow (since v0.5)

FmtLib is automatically fetched by CMake. 

TaskFlow is header-only library and most of it is included.

## Windows specific

OpenCV must be downloaded as exe file, which is self-extracting archive or compiled from sources.

OneTBB must be downloaded and extracted, possibly compiled from sources.

Editing CMakeLists.txt file is needed on Windows, as path to libraries are hardcoded there.

Since 2023-02-17, DLLs are copied to directory automatically. Finally!

Build was tried only in Visual Studio 2019 and 2022, it works in Visual Studio Code, but requires Visual Studio build tools.

## Specific for Linux or FreeBSD (and VirtualBox)


To install prerequisites run the following commands as root

| System | Command |
|---|---|
| FreeBSD | `pkg install cmake opencv onetbb` |
| Ubuntu  | `apt install cmake libtbb-dev libopencv-dev` |
| OpenSuse | `zypper install cmake tbb-devel opencv-devel` |

Then compile sources
```
$ cd reaction-diffusion-cpp
$ cmake -DCMAKE_BUILD_TYPE=Debug . -B build
$ cd build
$ make
$ ./reaction_diffusion
```

NOTE: CMAKE_BUILD_TYPE is optional. Can be `Debug` or `Release`

There are is a problem that AVX/FMA instruction sets are not supported in VirtualBox.
* `-march=x86-64-v3` program compiles and crashes, no matter if program has enabled instric instructions or not
* `-march=native` is ok, if AVX instructions are disabled (see `#define HAS_AVX` in the code)

# Performance notes

![speed.png](speed.png)

Performance tested on Ryzen 5900X (12C/24T) with 64MB RAM (2xKingston KF3200C16D4/32GX), usually on 1280x720 data with program compiled by MSVC 2022 Community Edition, gcc 9.4.0 and clang 12.0.0 (on Ubuntu 20.04.5 WSL)

On a single core, AVX code is roughly 5.5 times faster, but both TBB and TaskFlow have some overhead.

Updater2 uses different pointer arithmetic, which is faster in debug build (and clang)

Updater3 and Updater4 are fastest, but when more threads are used, program is likely limited by a memory bandwith.

Data are 32bit float and two images are updated in each iteration. One test batch consists of 2000 iterations. This gives 2000x2x4x1280x720 bytes processed per iteration - each batch processes 14.74GB of data (ten thousand floppy discs). Doing so in 560ms means data throughput of 26GB/s.
According to various sources (e.g. https://www.cpu-monkey.com/en/cpu-amd_ryzen_9_5900x) memory bandwidth is 48-56GB/s. Using just two CPU cores and AVX code is almost enough. 4 cores seem optimal. Above that slow algorithms have some benefits, fast ones are getting worse and difference between 3 and 24 threads negligible.

(Not so) surprisingly, GCC code is fastest, Clang is 2nd, Microsoft C++ compiler is the worst.

# ChangeLog

### v0.7

* Improvement in CMakeLists.txt
* Linux, FreeBSD compatibility
* Data throughput measurement

### v0.6

* 3rd updater slighly improved
* Documentation update
* Some changes in coefficients, different pattern

### v0.5 

* 4th updater, adds TaskFlow for parallelism (mostly for curiosity)

### v0.4

* Code switched to generate long video (30s from million iterations)

### v0.3

* 3rd updater, faster processing using AVX instruction set (close to memory throughput bottleneck, more threads won't help)

### v0.2

* 2nd updater, speed test

### v0.1

* Somewhat working version, generates video frames
