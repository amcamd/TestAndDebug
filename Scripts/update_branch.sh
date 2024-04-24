#!/bin/bash

function display_help()
{
cat << EOF
Script to update branch

  Usage:
    $0 <options>

  Options:
    -b|--branch        develop, master, staging                                  default: develop
    -r|--repository    rocBLAS-internal, rocBLAS, hipBLAS, Tensile, rocSOLVER    default: rocBLAS-internal
EOF
}

ORIGIN="amcamd"
UPSTREAM="ROCm"

BRANCH=develop
REPOSITORY=rocBLAS-internal

echo "BRANCH = $BRANCH"
echo "REPOSITORY = $REPOSITORY"

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      display_help
      exit 0
      ;;
    -b|--branch)
      BRANCH="$2"
      shift # past argument
      shift # past value
      ;;
    -r|--repository)
      REPOSITORY="$2"
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

echo "BRANCH = $BRANCH"
echo "REPOSITORY = $REPOSITORY"


#require branch to be one of: master, staging, develop
if [[ $BRANCH != "master" ]] && [[ $BRANCH != "staging" ]] && [[ $BRANCH != "develop" ]]; then
    echo "Usage: $0 -b <branch>"
    echo "where branch =  master  or  staging  or   develop"
    exit 1
fi

#require repository to be one of: rocBLAS-internal, rocBLAS, hipBLAS
if [[ $REPOSITORY != "rocBLAS-internal" ]] && [[ $REPOSITORY != "rocBLAS" ]] && [[ $REPOSITORY != "hipBLAS" ]] && [[ $REPOSITORY != "Tensile" ]] && [[ $REPOSITORY != "rocSOLVER" ]]; then
    echo "Usage: $0 -r <repository>"
    echo "where repository =  rocBLAS-internal  or  rocBLAS  or  hipBLAS"
    exit 1
fi

#prompt for confirmation to proceed
while true; do
    read -p "Confirm you want to update $ORIGIN:$REPOSITORY/$BRANCH?" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) echo "exiting script"; exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

#clone branch and rebase it on upstream/$BRANCH
mkdir -p ~/repos/rocBLASmergeDevelop
cd ~/repos/rocBLASmergeDevelop
rm -rf $REPOSITORY

git clone  git@github.com:$ORIGIN/$REPOSITORY.git
cd $REPOSITORY
git checkout $BRANCH

git remote add upstream git@github.com:$UPSTREAM/$REPOSITORY.git
git fetch upstream

git rebase upstream/$BRANCH $BRANCH
git status

#prompt for confirmation before pushing to origin/$BRANCH
while true; do
    read -p "Do you wish to push to origin:$REPOSITORY/$BRANCH?" yn
    case $yn in
        [Yy]* ) git push; break;;
        [Nn]* ) echo "change is in ~/repos/rocBLASmergeDevelop/$REPOSITORY"; exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

