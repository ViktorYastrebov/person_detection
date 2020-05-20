IF NOT EXIST ".\objects32d" mkdir objects32d
cd .\objects32d
cmake -D CMAKE_BUILD_TYPE=Debug -G"Visual Studio 15 2017" ..\