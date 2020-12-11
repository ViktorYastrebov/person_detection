#! /bin/sh

if [ ! -d "./objects64" ] 
then
#    echo "Directory /path/to/dir exists." 
  mkdir "./objects64" 
fi
cd "./objects64"
exec cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=ON ../

