REQUIREMENTS:

* OneTBB (Intel's Thread Building Blocks) https://github.com/oneapi-src/oneTBB/releases
* OpenCV https://opencv.org/releases/ (open exe as zip, unpack)

Editing CMakeLists.txt file is needed (i hardcoded my paths to libraries there)

These DLLs must be copied into directory with binary manually from
```
opencv-4.7.0\build\x64\vc16\bin 
oneapi-tbb-2021.7.0\redist\intel64\vc14\
```
to 
```
Release/opencv_world470.dll
Release/tbb12.dll
Debug/opencv_world470d.dll
Debug/tbb12_debug.dll
Debug/tbbmalloc_debug.dll
```