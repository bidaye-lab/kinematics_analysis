# Visualize joint coordinates in 3D
The 3D coordinates from Anipose can be visualized using the program [VMD](http://www.ks.uiuc.edu/Research/vmd/),
which is used in chemistry to explore the 3D structures of molecules.


## Generate XYZ files
The output from Anipose is the x, y, and z coordinates for each joint.
Each row in the resulting CSV file corresponds to one movie frame.
This data can be converted to a XYZ trajectory file,
which is in essence a concatenation of "molecular" structures.
VMD can play these trajectory files.

### ...from dataframe (notebook)
An example on how to generate XYZ files from a dataframe is given in 
[ball_fitting_example.py](../scripts/ball_fitting_example.py)

### ...from CSV (command line)
The python files in [`src/xyz_trajectory/`](../src/xyz_trajectory/) can be used as command line tools:
1. convert the CSV file to an XYZ file using `generate_xyz.py`:
    - activate DLC conda environment
    - navigate to folder containing CSV file
    - run `python generate_xyz.py 3Dpose.csv` \
    -OR- \
    - run `python generate_xyz.py 3Dpose.csv --ball 0.25 1.25 2.5` to also add the ball coordinates \
        note: `generate_xyz.py` has to be in the same folder or python path,
        alternatively specify full path to `generate_xyz.py`\
    -OR- \
    - run `python add_ball.py 3Dpose.csv` to add or update ball coordinates after the XYZ file was already generated
2. open with VMD (see below)
    - only if ball coordinates were added:\
    in the VMD terminal, type `ball 33` to adjust the diameter of the ball "atom" in VMD \
    note: `33` is the diameter of the ball in anipose coordinates times the scaling factor printed by `generate_xyz.py`


The scaling factor is introduced during the conversion from CSV to XYZ for a better representation in VMD.


## Changing the representation
All aspects of VMD can be controlled using the `tcl` scripting language.
The instructions on how to display the "fly molecule" are stored in the `vmd.rc` file (see below).
If you want to change anything about the default representation, it is convenient to edit this file, 
so any newly opened XYZ file will be displayed in the same way.

Note that on windows you need administrator access to edit the `vmd.rc` file at its deault location.
A convenient way would be to edit a copy of the file, say, on the desktop, and copy/paste the updated file confirming administrator access.

The best way to learn the correct commands that need to go into the `vmd.rc` file is to activate logging:
In the VMD main window, click `File` and `Log Tcl commands to console`. 
For any changes that you make through the GUI, the corresponding `tcl` command will appear in the console.

Note that all settings are specific for the "fly molecule"; any other actual molecule will most likely look very strange 
and you would need to restore the original `vmd.rc` file.
Some may argue that the fly looks strange, though.

When comparing multiple flies, it may be helpful to color each fly differently.
This can be done by typing `uni COLORID` into the VMD terminal, where `COLORID` is an integer.
This will apply the color the _top_ fly, which can be selected in the VMD main window.

## Playing trajectories and making movies
VMD can be used to view time-dependent, such as protein folding or chemical reactions 
and it can also render high-quality movies.
More resources can be found on the [VMD website](https://www.ks.uiuc.edu/Training/Tutorials/vmd/tutorial-html/node3.html).

To convert the trajectory into a movie, select "Extensions > Visualization > Movie Maker".
Here, chose "Renderer > Tachyon Internal".
Under "Movie Settings" select "Trajectory" and _uncheck_ "Delete image files".
Choose a suitable folder for the rendered images with "Set working direcory".
Now hit "Make movie", which may take a while, use all avaible CPU cores, and may require quite a bit of storage (~5 GB for 1400 frames).

Under windows, this will most likely end with a "Program not found" error (Could not locate videomach.exe).
Click "No" and for the following "Application Error" click "OK".
This is ok, because the individual image files are kept (see above) and can be converted into a movie using `ffmpeg`:
Open the anaconda prompt, activate the DLC conda environment and navigate to the directory containing the generated image files.
Run 
```
ffmpeg -framerate 50 -i untitled.%05d.bmp video.mp4
```
Here `video.mp4` is the name of the video file. The speed can be determined with the frame rate (original recordings: 200 Hz).
Now the image files can be deleted.

The resolution of the video can be adjusted by adjusting the size of the VMD window before rendering.
By default, ambient occlusion is turned on, which will create better 3D images.
However, this significantly increases the rendering time.

# Installation
Follow these instructions to install VMD and set it up to display the "fly molecule":
- download VMD 1.9.3 on the official [website](https://www.ks.uiuc.edu/Development/Download/download.cgi?PackageName=VMD)
  - this will require you to create an account (simply email and password)
- install
- replace the file [`vmd.rc`](vmd.rc) in the installation folder with the file from here
  - the default installation folder: `C:\Program Files (x86)\University of Illinois\VMD`
- associate XYZ files with VMD
  - right click on XYZ file (see above)
  - select "Open with..."
  - navigate to installation folder
  - select `vmd.exe`
