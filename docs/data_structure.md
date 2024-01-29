# Data structure

Leg kinematics data is stored in an HDF5 file and can be loaded with the `data_loader.load_data_hdf` function (see example notebooks),
which returns dictionary mapping genotype names to pandas dataframes.
Each row in the dataframe is single frame recorded at 200 Hz.
Each dataframe can contain multiple trials from multiple recording sessions.

|colunmn name|desciption|
|---|---|
|`flynum: int`|integer specifying specimen, 1-based|
|`trial: int`|integer specifying trial number, 1-based|
|`fnum: int`|integer specifying frame number, 0-based, resets for each `flynum` but not `tnum` |
|`SF: int`|stimulation frequency in given `tnum`, Hz|
|e.g. `R-F-ThC_x: float`|x coordinate of thorax-coxa joint in right front leg, a.u. (see below)|
|e.g. `R-WH_x: float`|x coordinate of right wing hinge, a.u. (see below)|
|e.g. `Notum_x: float`|x coordinate of the notum, a.u. (see below)|
|e.g. `L1A_abduct: float` | TODO | 
|e.g. `x_pos: float` | x ball position at each frame (see below) | 
|e.g. `x_vel: float` | x ball velocity at each frame (see below) |
|e.g. `R-F_stepcycle: bool` | indicating if right front leg is touching the surface (`True`) or not (`False`)|

The naming convention for columns positions is as follows:
- leg
    - `R`: right or `L`: left side
    - `F`: front `M`: mid `H`: hind leg|
- joint or point
    - `ThC`: thorax-coxa,
    - `CTr`: coxa-trochanter
    - `TiTa`: tibia-tarsus
    - `TaG`: tarsus-ground
    - `WH`: wing hinge
    - `Notum`: notum
- axis 
    - `x`, `y`, `z` coordinates of joint
    - `r` distance from center of ball (only available after ball fitting)

The naming convention for the angles is as follows:
- leg
    - `R`: right or `L`: left side
    - `1`: TODO
- joint
    - `A`: TODO
- angle
    - `abduct`: abduction
    - `flex`: flexion
    - `rot`: rotation

Important: The coordinate system for the points on the fly body is different than the coordinate system for the ball movement.

The ball tracker reports ball movement along the following directions:

![ball_tracking](ball_tracking.svg)
