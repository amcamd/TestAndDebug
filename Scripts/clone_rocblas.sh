#!/bin/bash

function display_help()
{
cat <<EOF
Script to clone rocBLAS

  Usage:
    $0 <options>

  Options:
    -r|--repository     rocBLAS_internal or rocBLAS       Default rocBLAS-internal)
    -o|--origin         amcamd or ROCmSoftwarePlatform    Default amcamd)
    -h|--help           this help information)
EOF
}

ORIGIN="amcamd"
REPOSITORY="rocBLAS-internal"

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      display_help
      exit 0
      ;;
    -r|--repository)
      REPOSITORY="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--origin)
      ORIGIN="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      echo "unknown argument"
      echo "Usage: $0 -b <branch> -r <repository>"
      echo "where branch =  master  or  staging  or   develop"
      echo "where repository == rocBLAS-internal, rocBLAS, hipBLAS"
      exit 1
      ;;
  esac
done

#require repository to be one of: rocBLAS-internal, rocBLAS
if [[ $REPOSITORY != "rocBLAS-internal" ]] && [[ $REPOSITORY != "rocBLAS" ]]; then
    echo "Usage: $0 -r <repository>"
    echo "where repository =  rocBLAS-internal  or  rocBLAS"
    exit 1
fi

#require origin to be one of: amcamd, ROCmSoftwarePlatform
if [[ $ORIGIN != "amcamd" ]] && [[ $ORIGIN != "ROCmSoftwarePlatform" ]]; then
    echo "Usage: $0 -r <origin>"
    echo "where origin = amcamd or ROCmSoftwarePlatform"
    exit 1
fi

echo "=====clone=rocblas============================================="
echo "==============================================================="

git clone  git@github.com:$ORIGIN/$REPOSITORY.git

if [[ $? -ne 0 ]]; then
    echo "clone error"
    exit 1
fi

cd $REPOSITORY
if [[ $? -ne 0 ]]; then
    echo "directory does not exist"
    exit 1
fi
