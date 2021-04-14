#!/bin/bash

#set -x

#sudo dpkg -r rocalution
#sudo dpkg -r rocblas
#export PATH=/opt/rocm/bin:$PATH

# docker commands
#sudo docker run --name andrew -tid --ipc=host --cap-add=SYS_PTRACE --privileged --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --group-add 44 -v /home:/home compute-artifactory.amd.com:5000/rocm-plus-docker/compute-rocm-rel-4.2:10-STG1 /bin/bash || exit 1
#
#sudo docker exec -e COLUMNS="`tput cols`" -e LINES="`tput lines`" -ti b3285621ba95 /bin/bash

echo "=====clone=rocblas============================================="
echo "==============================================================="
#git clone https://github.com/amcamd/rocBLAS.git 
git clone git@github.com:amcamd/rocBLAS-internal.git
#git clone git@github.com:ROCmSoftwarePlatform/rocBLAS-internal.git https://github.com/amcamd/rocBLAS-internal.git 
if [[ $? -ne 0 ]]; then
    echo "clone error"
    exit 1
fi

cd rocBLAS-internal
if [[ $? -ne 0 ]]; then
    echo "cd error"
    exit 1
fi

git checkout develop

echo "==============================================================="
echo "=====build=rocblas=with=install.sh============================="
echo "==============================================================="


export HIPCC_LINK_FLAGS_APPEND="-O3 -parallel-jobs=4"
export HIPCC_COMPILE_FLAGS_APPEND="-O3 -Wno-format-nonliteral -parallel-jobs=4"


if [ "$(/opt/rocm/bin/rocm_agent_enumerator | grep -m 1 gfx900)" == "gfx900" ]; then
    echo "=====ISA = gfx900, use -agfx900 directive ===================="
    time VERBOSE=1 ./install.sh -agfx900 -c 2>&1 | tee install.out
elif [ "$(/opt/rocm/bin/rocm_agent_enumerator | grep -m 1 gfx906)" == "gfx906" ]; then
    echo "=====ISA = gfx906, use -agfx906:xnack- directive ===================="
    time VERBOSE=1 ./install.sh -agfx906:xnack- -c 2>&1 | tee install.out
elif [ "$(/opt/rocm/bin/rocm_agent_enumerator | grep -m 1 gfx908)" == "gfx908" ]; then
    echo "=====ISA = gfx908, use -agfx908:xnack- directive ===================="
    time VERBOSE=1 ./install.sh -agfx908:xnack- -c 2>&1 | tee install.out
elif [ "$(/opt/rocm/bin/rocm_agent_enumerator | grep -m 1 gfx90a)" == "gfx908" ]; then
    echo "=====ISA = gfx908, use -agfx908:xnack- directive ===================="
    time VERBOSE=1 ./install.sh -agfx90a:xnack- -c 2>&1 | tee install.out
elif [ "$(/opt/rocm/bin/rocm_agent_enumerator | grep -m 1 gfx1010)" == "gfx908" ]; then
    echo "=====ISA = gfx908, use -agfx908:xnack- directive ===================="
    time VERBOSE=1 ./install.sh -agfx1010 -c 2>&1 | tee install.out
elif [ "$(/opt/rocm/bin/rocm_agent_enumerator | grep -m 1 gfx1012)" == "gfx908" ]; then
    echo "=====ISA = gfx908, use -agfx908:xnack- directive ===================="
    time VERBOSE=1 ./install.sh -agfx1012 -c 2>&1 | tee install.out
elif [ "$(/opt/rocm/bin/rocm_agent_enumerator | grep -m 1 gfx1030)" == "gfx908" ]; then
    echo "=====ISA = gfx908, use -agfx908:xnack- directive ===================="
    time VERBOSE=1 ./install.sh -agfx1030 -c 2>&1 | tee install.out
else
    echo "build fat binary, ISA != gfx900 and ISA != gfx906 and ISA != gfx908"
    time VERBOSE=1 ./install.sh -c 2>&1 | tee install.out
fi
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
LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocblas-test --gtest_filter=*quick* 
if [[ $? -ne 0 ]]; then
    echo "quick test error"
    exit 1
fi


echo "==============================================================="
echo "=====run=pre_checkin=tests====================================="
echo "==============================================================="
LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocblas-test --gtest_filter=*pre_checkin*
if [[ $? -ne 0 ]]; then
    echo "pre_checkin test error"
    exit 1
fi


echo "==============================================================="
echo "=====run=nightly=tests========================================="
echo "==============================================================="
LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocblas-test --gtest_filter=*nightly* 
if [[ $? -ne 0 ]]; then
    echo "nightly test error"
    exit 1
fi

