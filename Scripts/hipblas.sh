#!/bin/bash

#set -x

#sudo dpkg -r rocalution
#sudo dpkg -r rocblas
#export PATH=/opt/rocm/bin:$PATH

# docker commands
#sudo docker run --name andrew -tid --ipc=host --cap-add=SYS_PTRACE --privileged --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --group-add 44 -v /home:/home compute-artifactory.amd.com:5000/rocm-plus-docker/compute-rocm-rel-4.2:10-STG1 /bin/bash || exit 1
#
#sudo docker exec -e COLUMNS="`tput cols`" -e LINES="`tput lines`" -ti b3285621ba95 /bin/bash

echo "=====clone=hipblas============================================="
echo "==============================================================="
git clone git@github.com:amcamd/hipBLAS.git
if [[ $? -ne 0 ]]; then
    echo "clone error"
    exit 1
fi

NO_SOLVER=""
NO_SOLVER="--no-solver"

ROCBLAS_PATH=""
ROCBLAS_PATH="$PWD/rocBLAS-internal/build_tensile/release/rocblas-install"

if [[ ! -d $ROCBLAS_PATH ]]; then
    echo "ROCBLAS_PATH does not exist"
    echo $ROCBLAS_PATH
    exit 1
fi
ROCBLAS_PATH="--rocblas-path $ROCBLAS_PATH"

cd hipBLAS
if [[ $? -ne 0 ]]; then
    echo "cd error"
    exit 1
fi

git checkout develop

echo "==============================================================="
echo "=====build=hipblas=with=install.sh============================="
echo "==============================================================="

export HIPCC_LINK_FLAGS_APPEND="-O3 -parallel-jobs=12"
export HIPCC_COMPILE_FLAGS_APPEND="-O3 -Wno-format-nonliteral -parallel-jobs=12"

time VERBOSE=1 ./install.sh $NO_SOLVER $ROCBLAS_PATH -c 2>&1 | tee install.out

if [[ $? -ne 0 ]]; then
    echo "install error"
    exit 1
fi

cd build/release/clients/staging
if [[ $? -ne 0 ]]; then
    echo "cd staging error"
    exit 1
fi

echo "==============================================================="
echo "=====run=quick=tests==========================================="
echo "==============================================================="
LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./hipblas-test --gtest_filter=*quick* 
if [[ $? -ne 0 ]]; then
    echo "quick test error"
    exit 1
fi


echo "==============================================================="
echo "=====run=pre_checkin=tests====================================="
echo "==============================================================="
LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./hipblas-test --gtest_filter=*pre_checkin*
if [[ $? -ne 0 ]]; then
    echo "pre_checkin test error"
    exit 1
fi


echo "==============================================================="
echo "=====run=nightly=tests========================================="
echo "==============================================================="
LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./hipblas-test --gtest_filter=*nightly* 
if [[ $? -ne 0 ]]; then
    echo "nightly test error"
    exit 1
fi

