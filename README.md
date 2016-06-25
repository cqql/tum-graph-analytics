# Large-Scale Graph Analytics with Bösen

## Installation

```sh
# Clone bosen
git clone https://github.com:petuum/bosen.git libs/bosen

# Fix bosen's CMakeLists.txt and modules so that we can include it
(cd libs/bosen; git apply ../../bosen-patches)
```

### Ubuntu

```sh
sudo apt-get -y install libgoogle-glog-dev libzmq3-dev libyaml-cpp-dev \
  libgoogle-perftools-dev libsnappy-dev libsparsehash-dev libgflags-dev \
  libboost-system1.55-dev libboost-thread1.55-dev libleveldb-dev \
  libconfig++-dev libghc-hashtables-dev libtcmalloc-minimal4 \
  libevent-pthreads-2.0-5 libeigen3-dev
```

### Arch Linux

```sh
sudo pacman -S snappy gperftools sparsehash leveldb gflags eigen libconfig yaml-cpp
```

## DMLC

The old DMLC code has been moved to the `dmlc` subdirectory.
