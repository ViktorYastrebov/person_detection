IF NOT EXIST ".\objects64d" mkdir objects64d
cd .\objects64d
cmake -D CMAKE_BUILD_TYPE=Debug -G"Visual Studio 15 2017 Win64" ..\