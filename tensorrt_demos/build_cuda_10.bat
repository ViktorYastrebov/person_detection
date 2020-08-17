IF DEFINED CUDA_PATH_V10_0 SET CUDA_PATH=%CUDA_PATH_V10_0%
IF NOT EXIST ".\objects64_10" mkdir objects64_10
cd .\objects64_10
cmake -D CMAKE_BUILD_TYPE=Release -G"Visual Studio 15 2017 Win64" ..\