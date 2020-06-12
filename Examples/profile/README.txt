

On windows or linux you'll need chrome or chromium installed respectively.

Then profiling with a command like pasted below generates a bunch of files in the current directory, then one you will later use is results.json.  I usually mv results.* into a folder to organize it.

/opt/rocm-3.5.0-12/bin/rocprof  --obj-tracking on --hip-trace --timestamp --stats ./rocblas-bench -f dot -r d -n 10000002

Run the web browser chrome/chromium and enter in the address bar:  
chrome://tracing/
There will be a "Load" button on the displayed page, hit and browser for output "results.json" and Okay.

This may or may not display much that is visible, the time extent is not properly encoded within the .json file. 
Hit the "?" button to see the navigation modes, if you are lucky the zoom will be good but I rarely am lucky.
I find it best to use the 'w' and 's' keys to zoom in on the cursor location, sometimes the time is just a sliver of color so put your mouse on that sliver and hold down 'w' to zoom in and see the timeline.

