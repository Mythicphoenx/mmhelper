# Introduction 

This project aims to provide an automated framework to detect, track, and analyse bacteria in a "Mother Machine" microfluidic setup.

# Prerequisites
In order to run momanalysis you must ensure that you have python installed on your computer.  
For windows we recommend either Anaconda or WinPython  
For Mac, Unix or Linux we recommend Anaconda  

WinPython:  https://winpython.github.io  
Anaconda:   https://conda.io/docs/user-guide/install/index.html

# Installing momanalysis

Installing momanalysis is simple. 

Just open Terminal, Anaconda prompt or WinPython command prompt and type the following:

> pip install -U "momanalysis link here!!!"

# Starting momanalysis

Once the package is installed, you can start the user interface by typing:

> momanalysis 

![Alt text](/docs/momanalysis_gui.png "User Interface")

# Running momanalysis

Running momanalysis is easy. You can manually add the file(s)/folder path, select it using the button provided or simple "drag and drop".  
The various input options and the respective parameters required on the user interface are listed below.  
N.B. If running in batch mode on a folder of images (can include fluorescence) then see the next section

No fluorescence  
* Single brightfield or phase contrast image - add the file and hit "start analysis"
* stack of brightfield or phase contrast images - add the file and hit "start analysis"

Fluorescence  
* Alternating stack of brightfield/phase contrast images and their matching fluorescent images (e.g. BF/Fluo/BF/FLuo) - add the file, select "combined fluorescence" and hit "start analysis"
* Corresponding fluorescent image(s) are in a separate file (stacks if multiple frames) - add the files, select "separate fluorescence" and hit "start analysis"

# Folder (Batch mode)
momanalysis also has the capacity to run in batch mode on a data set containing multiple areas. In order to do this the files within the folder must be in the following format:  
"a_b_c.tif" where:
* a = an identifier for the area in which the image was taken. E.g. "01" for all images in the first area of the mother machine, "02" for the second, etc.
* b = a time stamp - this allows momanalysis to stack images from the same timepoint in chronological order
* c = any further identifiers of your choice

Once you have a suitable folder, you simply add it to the interface as described and select "batch run".  
If it is just brightfield/phase then you can hit "start analysis", but if your folder includes fluorescent images (see note below) then select "combined fluorescence" before starting the analysis  
  
Note: When including fluorescent images in a folder, the timestamps should alternate between BF/Phase and the matching fluorescent image (like alternating stack above)

# Additional options
You also have the  option of specifying where you want the output of the analysis to be saved. If this is undefined then it will be in the same folder as the input
