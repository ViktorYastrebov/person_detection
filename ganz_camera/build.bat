IF NOT EXIST ".\objects64" mkdir objects64
cd .\objects64
cmake -D CMAKE_BUILD_TYPE=Release -G"Visual Studio 15 2017 Win64" ..\