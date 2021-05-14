#!/bin/bash

BRANCH="not_initialized"
ORIGIN="amcamd"
UPSTREAM="ROCmSoftwarePlatform"

#read command line arguments
#TODO add ORIGIN and UPSTREAM forks
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -b|--branch)
    BRANCH="$2"
    shift
    shift
    ;;
    *)
    echo "Usage: $0 -b <branch>"
    echo "where branch =  master  or  staging  or   develop"
    exit 1
    ;;
esac
done

#require branch to be one of: master, staging, develop
if [[ $BRANCH != "master" ]] && [[ $BRANCH != "staging" ]] && [[ $BRANCH != "develop" ]]; then
    echo "Usage: $0 -b <branch>"
    echo "where branch =  master  or  staging  or   develop"
    exit 1
fi

#prompt for confirmation to proceed
while true; do
    read -p "Confirm you want to update $ORIGIN/$BRANCH?" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) echo "exiting script"; exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

#clone branch and rebase it on upstream/$BRANCH
mkdir -p ~/repos/rocBLASmergeDevelop
cd ~/repos/rocBLASmergeDevelop
rm -rf rocBLAS-internal

git clone  git@github.com:$ORIGIN/rocBLAS-internal.git
cd rocBLAS-internal
git checkout $BRANCH

git remote add upstream git@github.com:$UPSTREAM/rocBLAS-internal.git
git fetch upstream

git rebase upstream/$BRANCH $BRANCH
git status

#prompt for confirmation before pushing to origin/$BRANCH
while true; do
    read -p "Do you wish to push to origin/$BRANCH?" yn
    case $yn in
        [Yy]* ) git push; break;;
        [Nn]* ) echo "change is in ~/repos/rocBLASmergeDevelop/rocBLAS-internal"; exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

