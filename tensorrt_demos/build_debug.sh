#! /bin/sh

if [ ! -d "./objects64d" ] 
then
#    echo "Directory /path/to/dir exists." 
  mkdir "./objects64d" 
fi
cd "./objects64d"
exec cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_VERBOSE_MAKEFILE=ON ../

