#!/bin/bash

if [ "$1" ]
  then
    VERSION="$1"
else
    VERSION="v0.14.0"
fi

read -p "Are you in the correct python environment? (y/n) " answer
case ${answer:0:1} in
    y|Y )
        ;;
    * )
        exit;;
esac

echo
echo Installing...
echo

clean_up() {
  test -d "$tmp_dir" && rm -rf "$tmp_dir"
}

tmp_dir=$( mktemp -d -t install-openmc.XXX)
trap "clean_up $tmp_dir" EXIT

set -euxo pipefail

cd $tmp_dir

sudo apt install g++ cmake libhdf5-dev libpng-dev libopenmpi-dev ninja-build -y

git clone --recurse-submodules https://github.com/openmc-dev/openmc.git
cd openmc
git checkout $VERSION
mkdir build && cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX=$HOME/.local -DOPENMC_USE_MPI=ON ..
ninja install

cd ..
pip install .

echo Finished
