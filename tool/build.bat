where cl || call "C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
dir build || mkdir build
set EXTRA_OPTIONS=-g
set EXTRA_OPTIONS=-O3
nvcc optix/launch.cu ^
  -I"C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.6.0\include" ^
  -I. ^
  --ptx ^
  -o data/launch.ptx %EXTRA_OPTIONS% && ^
wsl -e xxd -i data/launch.ptx > data/launch_ptx.h && ^
nvcc main.cu deps/lodepng/lodepng.cpp ^
  -DUSE_OPTIX ^
  -I"C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.6.0\include" ^
  -I. ^
  --extended-lambda --expt-relaxed-constexpr --std=c++17 -lnvrtc ^
  -o main.exe %EXTRA_OPTIONS% && ^
echo main
