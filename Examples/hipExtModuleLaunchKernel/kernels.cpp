/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "hip/hip_runtime.h"

extern "C" __global__ void work_kernel(float* a, float* b) {
    int tx = hipThreadIdx_x;
    // only purpose of below calculations is to take enough time relative to launch time
    // that any-order launch can demonstrate performance improvement
    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 


    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 

    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 
    b[tx] += (b[tx] / 0.555 + 0.1234) / (a[tx] / 0.333 + 0.2345); 
    a[tx] += (b[tx] / 0.444 + 0.6789) / (a[tx] / 0.666 + 0.9876); 
}


extern "C" __global__ void empty_kernel() {
}
