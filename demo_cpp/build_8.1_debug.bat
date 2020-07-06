IF NOT EXIST ".\objects64d_win8.1" mkdir objects64d_win8.1
cd .\objects64d_win8.1
cmake -D CMAKE_BUILD_TYPE=Debug -G"Visual Studio 15 2017 Win64" ..\