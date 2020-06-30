IF NOT EXIST ".\objects64_win8.1" mkdir objects64_win8.1
cd .\objects64_win8.1
cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_SYSTEM_VERSION=8.1 -G"Visual Studio 15 2017 Win64" ..\