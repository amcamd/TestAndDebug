#!/bin/bash

function display_help()
{
cat <<EOF
Script to clone mono-repo, rocBLAS-internal, rocBLAS, or hipBLAS

  Usage:
    $0 <options>

  Options:
    -r|--repository     mono-repo, rocBLAS_internal, rocBLAS, hipBLAS      Default mono-repo)
    -o|--origin         amcamd or ROCm    Default amcamd)
    -c|--connection     ssh or https                      Default ssh
    -h|--help           this help information)
EOF
}

ORIGIN="ROCm"
REPOSITORY="mono-repo"
CONNECTION="ssh"

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
    -c|--connection)
      CONNECTION="$2"
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
      echo "where repository == mono-repo, rocBLAS-internal, rocBLAS, hipBLAS"
      exit 1
      ;;
  esac
done

#require repository to be one of: mono-reop, rocBLAS-internal, rocBLAS, hipBLAS
if [[ $REPOSITORY != "mono-repo" ]] && [[ $REPOSITORY != "rocBLAS-internal" ]] && [[ $REPOSITORY != "rocBLAS" ]] && [[ $REPOSITORY != "hipBLAS" ]]; then
    echo "Usage: $0 -r <repository>"
    echo "where repository =  mono-repo, rocBLAS-internal, rocBLAS, or hipBLAS"
    exit 1
fi

#require origin to be one of: amcamd, ROCm
if [[ $ORIGIN != "amcamd" ]] && [[ $ORIGIN != "ROCm" ]]; then
    echo "Usage: $0 -r <origin>"
    echo "where origin = amcamd or ROCm"
    exit 1
fi

echo "=====clone=rocblas============================================="
echo "==============================================================="

if [[ $CONNECTION == "https" ]]; then
    cat ~/githubToken
    git clone https://github.com/$ORIGIN/$REPOSITORY.git
else

    if [[ $ORIGIN == "ROCm" ]] && [[ $REPOSITORY == "mono-repo" ]]; then
        git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
        cd rocm-libraries
        git sparse-checkout init --cone
        git sparse-checkout set projects/rocblas shared/tensile
        git checkout develop
    else
        git clone  git@github.com:$ORIGIN/$REPOSITORY.git
    fi
fi

if [[ $? -ne 0 ]]; then
    echo "clone error"
    exit 1
fi

if [[ $ORIGIN == "ROCm" ]] && [[ $REPOSITORY == "mono-repo" ]]; then
    cd projects/rocblas
else
    cd $REPOSITORY
fi

if [[ $? -ne 0 ]]; then
    echo "directory does not exist"
    exit 1
fi
