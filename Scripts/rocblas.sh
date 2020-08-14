#!/bin/bash

#set -x

#sudo dpkg -r rocalution
#sudo dpkg -r rocblas
#export PATH=/opt/rocm/bin:$PATH

echo "=====clone=rocblas============================================="
echo "==============================================================="
git clone https://github.com/amcamd/rocBLAS-internal.git 
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
if [ "$(/opt/rocm/bin/rocm_agent_enumerator | grep -m 1 gfx900)" == "gfx900" ]; then
    echo "=====ISA = gfx900, use -agfx900 directive ===================="
    time ./install.sh -agfx900 -c
elif [ "$(/opt/rocm/bin/rocm_agent_enumerator | grep -m 1 gfx906)" == "gfx906" ]; then
    echo "=====ISA = gfx906, use -agfx906 directive ===================="
    time ./install.sh -agfx906 -c
elif [ "$(/opt/rocm/bin/rocm_agent_enumerator | grep -m 1 gfx908)" == "gfx908" ]; then
    echo "=====ISA = gfx908, use -agfx908 directive ===================="
    time ./install.sh -agfx908 -c
else
    echo "build fat binary, ISA != gfx900 and ISA != gfx906 and ISA != gfx908"
    time ./install.sh -c
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

