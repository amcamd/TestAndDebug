#!/usr/bin/python3

usage = \
''' A script to check if a commit in a given commit on a ROCm library
is in a specified dkms-no-npi build.

This script must be run from a checked out copy of the ROCm library in
question.

Usage:
./find-no-npi.py 
\t\t\t-h         \tdisplay Display help message
\t\t\t-l (string)\tspecify ROCm library (default: rocBLAS)
\t\t\t-b (int)   \tspecify Specify build number (default: search all builds)
\t\t\t-H (int)   \tspecify Specify library commit hash
'''

import sys, getopt
import codecs

# Get the last succesful build number by querying the URL.
def getlastsuccesflbuild():
    cmd = ["curl","http://rocm-ci.amd.com/job/compute-rocm-dkms-no-npi-hipclang/lastSuccessfulBuild/buildNumber"]
    import subprocess, os
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=os.environ.copy())
    proc.wait()
    rc = proc.returncode
    if(rc == 0):
        out, err = proc.communicate()
        return int(out.decode().strip())

# Get the oldest avilable build number by searching for the first build that doesn't 404.
def getoldestsuccesflbuild():
    dkmirev = getlastsuccesflbuild()
    url = "http://rocm-ci.amd.com/job/compute-rocm-dkms-no-npi-hipclang/" + str(dkmirev) + "/artifact/manifest.xml"
    import requests
    goodrev = dkmirev
    badrev = dkmirev
    stepsize = 1
    while requests.get(url).status_code != 404 and dkmirev > 0 :
        goodrev = dkmirev
        dkmirev -= stepsize
        stepsize *= 2
        url = "http://rocm-ci.amd.com/job/compute-rocm-dkms-no-npi-hipclang/" + str(dkmirev) + "/artifact/manifest.xml"
    else:
        badrev = dkmirev
    chrekrev = None
    while goodrev - badrev > 1:
        checkrev = (goodrev + badrev + 1) // 2
        url = "http://rocm-ci.amd.com/job/compute-rocm-dkms-no-npi-hipclang/" + str(checkrev) + "/artifact/manifest.xml"
        if requests.get(url).status_code == 404:
            badrev = checkrev
        else:
            goodrev = checkrev
    return checkrev + 1

# Get the hash for the version of the given project from the manifest at the given url.
def gethashfromurl(url, projname):
    import requests
    r = requests.get(url)
    if r.status_code != 200:
        print("url fetch failed for " + url)
        sys.exit(1)
    xmldat = r.text
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xmldat)
    for child in root:
        #print(child.get('name'))
        if child.get('name') != None and child.get('name').startswith(projname):
            return(child.get('revision'))

# Check that the cwd contains a git repo which include the selected hash
def repohashash(hash2check):
    cmd = ["git", "branch", "-r", "--contains", hash2check]
    print(cmd)
    import subprocess, os, tempfile
    fout = tempfile.TemporaryFile(mode="w+")
    ferr = tempfile.TemporaryFile(mode="w+")
    proc = subprocess.Popen(cmd, env=os.environ.copy(),
                            stdout=fout, stderr=ferr)
    proc.wait()
    rc = proc.returncode
    return rc == 0
        
# Determine if hash1 is an ancestor of hash 2
def hashis(hash1, hash2):
    import subprocess, os, tempfile
    #cmd = ["git", "merge-base", "--is-ancestor", hash1, hash2]
    cmd = ["git", "merge-base", hash1, hash2]
    print(cmd)
    import subprocess, os
    proc = subprocess.Popen(cmd, env=os.environ.copy(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = proc.communicate() 
    for line in output.splitlines():
        val = line.decode('utf-8').strip()
        print( f'********* val= {val}')
        break
    else:
        val = hash2
    print(f'****** val {val} hash2 {hash2}')
    rc = proc.returncode
    return val == hash2 

# Find the first build that contains the given hash for the project.
def findfirstbuildwithhash(rev0, rev1, hash2check, projname):
    goodrevs = []
    for i in range(rev0, rev1):
        goodrevs.append(i)
    ilow = 0
    ihigh = len(goodrevs) - 1
    irev = 0
    # Find the commit in between the builds:
    while ihigh - ilow > 1:
        irev = (ilow + ihigh + 1) // 2
        #print([ihigh, ilow, irev, goodrevs[irev]])
        url = "http://rocm-ci.amd.com/job/compute-rocm-dkms-no-npi-hipclang/" + str(goodrevs[irev]) + "/artifact/manifest.xml"
        import requests
        if requests.get(url).status_code != 200:
            # Sometimes we get an internal server error, so we need to skip certain data points.
            print("failed to retrieve " + url)
            del goodrevs[irev]
            ihigh -= 1
            continue
        manifestversion = gethashfromurl(url, projname)
        if manifestversion == None:
            print("failed to retrieve " + url)
            sys.exit(1)
        #print(manifestversion)
        rc = hashis(hash2check, manifestversion)
        if rc != 0:
            ilow = irev
        else:
            ihigh = irev
    return goodrevs[irev + 1]

# Get all available builds and search for the first build with the given commit for the project.
def checkallbuildsforhash(hash2check, projname):
    import requests
    
    rev0 = getoldestsuccesflbuild()
    print("First available build:  " + str(rev0))
    
    rev1 = getlastsuccesflbuild()
    print("Latest available build: " + str(rev1))

    # Check if the latest build has the commit:
    url = "http://rocm-ci.amd.com/job/compute-rocm-dkms-no-npi-hipclang/" + str(rev1) + "/artifact/manifest.xml"
    rc = requests.get(url).status_code
    if rc != 200:
        print(rc)
    manifestversion = gethashfromurl(url, projname)
    if manifestversion == None:
        print("Failed to get " + url)
        sys.exit(1)
    rc = hashis(hash2check, manifestversion)
    if rc != 0:
        print("commit not present in dkms build")
        return

    # Check if the earliest build has the commit:
    url = "http://rocm-ci.amd.com/job/compute-rocm-dkms-no-npi-hipclang/" + str(rev0) + "/artifact/manifest.xml"
    manifestversion = gethashfromurl(url, projname)
    rc = hashis(hash2check, manifestversion)
    if rc == 0:
        print("Commit present in first available build")
        return
        
    first = findfirstbuildwithhash(rev0, rev1, hash2check, projname)
    print("Commit first appears in build " + str(first))


# Check an individual build for a commit in a given project
def checkbuildforhash(dkmirev, hash2check, projname):
    print("Checking build", str(dkmirev))

    url = "http://rocm-ci.amd.com/job/compute-rocm-dkms-no-npi-hipclang/" + str(dkmirev) + "/artifact/manifest.xml"
    manifestversion = gethashfromurl(url, projname)
    if manifestversion == None:
        print("Build not available")
        sys.exit(0)

    print(projname, "version in build", dkmirev, ":", manifestversion)

    rc = hashis(hash2check, manifestversion)
    if(rc == 0):
        print("It's there!")
    else:
        print("It's not there!")

def main(argv):
    projname = "rocBLAS"
    dkmirev = None
    hash2check = "bdb82c68cb8756a0a82f0d1405e3b3e8e58f5c5f"
    
    try:
        opts, args = getopt.getopt(argv,"hl:b:H:")
    except getopt.GetoptError:
        print("error in parsing arguments.")
        print(usage)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h"):
            print(usage)
            exit(0)
        elif opt in ("-l"):
            projname = arg
        elif opt in ("-b"):
            dkmirev = arg
        elif opt in ("-H"):
            hash2check = arg

    print("Searching for commit " + hash2check + " in project " + projname)

    if not repohashash(hash2check):
        print("working directory does not contain a git repo with commit " + hash2check)
        sys.exit(1)
    
    if dkmirev == None:
        checkallbuildsforhash(hash2check, projname)
    else:
        checkbuildforhash(dkmirev, hash2check, projname)
        
if __name__ == "__main__":
    main(sys.argv[1:])

