IF DEFINED CUDA_PATH_V10_0 SET CUDA_PATH=%CUDA_PATH_V10_0%
IF NOT EXIST ".\objects64d" mkdir objects64d
cd .\objects64d
cmake -D CMAKE_BUILD_TYPE=Debug -G"Visual Studio 15 2017 Win64" ..\