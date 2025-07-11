#!/bin/bash

function display_help()
{
cat <<EOF
Script to build rocBLAS and run tests

  Usage:
    $0 <options>

  Options:
    -r|--rocBLAS        mono-repo, rocBLAS_internal or rocBLAS  Default mono-repo)
    -t|--tensile        tensile or no_tensile        Default tensile)
    -l|--hipblaslt      hipblaslt or no_hipblaslg    Default hipblaslt)
    -b|--branch         develop, master, ...         Default develop)
    -q|--quick          false or true                Default false)
    -p|--precheckin     false or true                Default false)
    -n|--nightly        false or true                Default false)
EOF
}

ROCBLAS=mono-repo
TENSILE=tensile
HIPBLASLT=hipblaslt
BRANCH=develop

QUICK=false
PRECHECKIN=false
NIGHTLY=false

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      display_help
      exit 0
      ;;
    -r|--rocBLAS)
      ROCBLAS="$2"
      shift # past argument
      shift # past value
      ;;
    -t|--tensile)
      TENSILE="$2"
      shift # past argument
      shift # past value
      ;;
    -l|--hipblaslt)
      HIPBLASLT="$2"
      shift # past argument
      shift # past value
      ;;
    -b|--branch)
      BRANCH="$2"
      shift # past argument
      shift # past value
      ;;
    -q|--quick)
      QUICK="true"
      shift # past argument
      ;;
    -p|--precheckin)
      PRECHECKIN="true"
      shift # past argument
      ;;
    -n|--nightly)
      NIGHTLY="true"
      shift # past argument
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      echo "unknown argument"
      exit 1
      ;;
  esac
done

if [[ $ROCBLAS != "mono-repo" && $ROCBLAS != "rocBLAS" && $ROCBLAS != "rocBLAS-internal" ]]; then
    echo "Usage: $0 -r <repository>" 
    echo "where repository is mono-repo or rocBLAS or rocBLAS-internal"
    exit 1
fi
if [[ $TENSILE != "tensile" && $TENSILE != "no_tensile" ]]; then
    echo "Usage: $0 -t <tensile>" 
    echo "where <tensile> is tensile  or  no_tensile"
    exit 1
fi
if [[ $HIPBLASLT != "hipblaslt" && $HIPBLASLT != "no_hipblaslt" ]]; then
    echo "Usage: $0 -t <hipblaslt>" 
    echo "where <hipblaslt> is hipblaslt  or  no_hipblaslt"
    exit 1
fi

if [[ $TENSILE == "tensile" ]]; then
    BUILD_TENSILE=""
    BUILD_DIR="build_tensile"
else
    BUILD_TENSILE="--no_tensile"
    BUILD_DIR="build_no_tensile"
fi

if [[ $HIPBLASLT == "hipblaslt" ]]; then
    BUILD_HIPBLASLT=""
else
    BUILD_HIPBLASLT="--no_hipblaslt"
    BUILD_DIR="build_no_hipblaslt"
fi

echo "rocBLAS    = $ROCBLAS"
echo "TENSILE    = $TENSILE"
echo "HIPBLASLT  = $HIPBLASLT"
echo "BRANCH     = $BRANCH"
echo "QUICK      = $QUICK"
echo "PRECHECKIN = $PRECHECKIN"
echo "NIGHTLY    = $NIGHTLY"

while true; do
    read -p "Confirm you want to build and run the tests above?" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) echo "exiting script"; exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

if [[ $ROCBLAS == "mono-repo" ]]; then
    cd rocm-libraries/projects/rocblas
else
    cd $ROCBLAS
fi
if [[ $? -ne 0 ]]; then
    echo "cd error"
    exit 1
fi

git checkout $BRANCH

echo "==============================================================="
echo "=====build=rocblas=with=install.sh============================="
echo "==============================================================="

export HIPCC_LINK_FLAGS_APPEND="-O3 -parallel-jobs=12"
export HIPCC_COMPILE_FLAGS_APPEND="-O3 -Wno-format-nonliteral -parallel-jobs=12"

if [ "$(/opt/rocm/bin/rocm_agent_enumerator | grep -m 1 gfx900)" == "gfx900" ]; then
    echo "=====ISA = gfx900, use -agfx900 directive ===================="
    ISA="-agfx900"
elif [ "$(/opt/rocm/bin/rocm_agent_enumerator | grep -m 1 gfx906)" == "gfx906" ]; then
    echo "=====ISA = gfx906, use -agfx906:xnack- directive ===================="
    ISA="-agfx906:xnack-"
elif [ "$(/opt/rocm/bin/rocm_agent_enumerator | grep -m 1 gfx908)" == "gfx908" ]; then
    echo "=====ISA = gfx908, use -agfx908:xnack- directive ===================="
    ISA="-agfx908:xnack-"
elif [ "$(/opt/rocm/bin/rocm_agent_enumerator | grep -m 1 gfx90a)" == "gfx90a" ]; then
    echo "=====ISA = gfx90a, use -agfx90a:xnack- directive ===================="
    ISA="-agfx90a:xnack-"
elif [ "$(/opt/rocm/bin/rocm_agent_enumerator | grep -m 1 gfx940)" == "gfx940" ]; then
    echo "=====ISA = gfx940, use -agfx940 directive ===================="
    ISA="-agfx940"
elif [ "$(/opt/rocm/bin/rocm_agent_enumerator | grep -m 1 gfx941)" == "gfx941" ]; then
    echo "=====ISA = gfx941, use -agfx941 directive ===================="
    ISA="-agfx941"
elif [ "$(/opt/rocm/bin/rocm_agent_enumerator | grep -m 1 gfx942)" == "gfx942" ]; then
    echo "=====ISA = gfx942, use -agfx942 directive ===================="
    ISA="-agfx942"
elif [ "$(/opt/rocm/bin/rocm_agent_enumerator | grep -m 1 gfx950)" == "gfx950" ]; then
    echo "=====ISA = gfx950, use -agfx950 directive ===================="
    ISA="-agfx950"
elif [ "$(/opt/rocm/bin/rocm_agent_enumerator | grep -m 1 gfx1010)" == "gfx1010" ]; then
    echo "=====ISA = gfx1010, use -agfx1010:xnack- directive ===================="
    ISA="-agfx1010"
elif [ "$(/opt/rocm/bin/rocm_agent_enumerator | grep -m 1 gfx1012)" == "gfx1012" ]; then
    echo "=====ISA = gfx1012, use -agfx1012:xnack- directive ===================="
    ISA="-agfx1012"
elif [ "$(/opt/rocm/bin/rocm_agent_enumerator | grep -m 1 gfx1030)" == "gfx1030" ]; then
    echo "=====ISA = gfx1030, use -agfx1030:xnack- directive ===================="
    ISA="-agfx1030"
else
    echo "build fat binary, ISA != gfx900 and ISA != gfx906 and ISA != gfx908"
    ISA=""
fi

#export ROCM_PATH=/opt/rocm-5.1.1/    # if ROCm not in /opt/rocm

time VERBOSE=1 ./install.sh $ISA $BUILD_TENSILE $BUILD_HIPBLASLT --build_dir $BUILD_DIR -cd --cmake_install  2>&1 | tee install.out

if [[ $? -ne 0 ]]; then
    echo "install error"
    exit 1
fi

cd $BUILD_DIR/release/clients/staging
if [[ $? -ne 0 ]]; then
    echo "cd staging error"
    exit 1
fi

if [[ $QUICK == "true" ]]; then
    echo "==============================================================="
    echo "=====run=quick=tests==========================================="
    echo "==============================================================="
    LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocblas-test --gtest_filter=*quick* 
    if [[ $? -ne 0 ]]; then
        echo "quick test error"
        exit 1
    fi
fi

if [[ $PRECHECKIN == "true" ]]; then
    echo "==============================================================="
    echo "=====run=pre_checkin=tests====================================="
    echo "==============================================================="
    LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocblas-test --gtest_filter=*pre_checkin*
    if [[ $? -ne 0 ]]; then
        echo "pre_checkin test error"
        exit 1
    fi
fi

if [[ $NIGHTLY == "true" ]]; then
    echo "==============================================================="
    echo "=====run=nightly=tests========================================="
    echo "==============================================================="
    LD_LIBRARY_PATH=/opt/rocm/hcc/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocblas-test --gtest_filter=*nightly* 
    if [[ $? -ne 0 ]]; then
        echo "nightly test error"
        exit 1
    fi
fi
