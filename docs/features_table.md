# Features Table

From the raw data structure containing the position and joint angle time traces, several stepping parameters are extracted to parameterize the gait. 
The extracted parameters depend on the step cycle precition or detection of peaks in the joint angle time traces.
To make the resulting data structure easy to use for regression analysis, all the features are binned over time intervals. The bin size should be specified such that each bin has atleast one full step cycle. The ideal bin size may vary across different genotypes depending on the speed of walking induced, but the same bin size should be used for all genotypes  being compared in any given model. 

|colunmn name|desciption|
|---|---|
|`flynum: int`|integer specifying specimen, 1-based|
|`trial: int`|integer specifying trial number, 1-based|
|`mean_x_vel: float`|mean forward velocity of the ball in the current bin. Absolute value used for averaging (similar columns exist for y and z velocities)|
|e.g. `L1_step_period: float`|Mean step period (unit: frames) (swing duration + stance duration) for the current bin (Nan value assigned if no step was initiated or the step in progress was not completed within in the bin; see below)|
|e.g. `L1_swing_dur: float`|Mean swing duration (unit: frames) for the current bin (Nan values asigned similar to above)|
|e.g. `L1_stance_dur: float` | Mean stance duration (unit: frames) for the current bin (Nan values asigned similar to above) | 
|e.g. `L1A_flex_count: float` | Number of peaks detected in the joint angle trace in the current bin | 
|e.g. `L1A_flex_heights: float` |Mean height of the peaks (unit: degrees) for the joint angle in the current bin | 
|e.g. `L1A_flex_widths: float` | Mean width of the peaks (unit: frames) for the joint angle in the current bin | 
|e.g. `L1A_flex_prominences: float` | Mean prominence of the peaks (unit: degrees) for the joint angle in the current bin (see prominence definition as used by scipy) | 



The naming convention for columns is as follows:
- leg
    - `R`: right or `L`: left side
    - `1`: front `2`: mid `3`: hind leg|

On choosing the bin size and 'NaN' values in columns:
- Note that the columns could contain Nan values when the corresponding events were not detected in the current bin. The occurance of Nan values can be minimized by making sure that the chosne bin size contians at least one full step (1 swing + 1 stance) and ideally two full steps. 
- How to handle the Nan values best is not established yet. Nan values are tricky because depending on the bin size, they may equally repsent a completely quiscent leg or the fact that the chosen bin size is too small and as a result most bins are not large enough to span an entire step.
- By choosing a very large bin size, Nan values may be fully eliminated, but at the same time all the features and velocities get smoothed, losing out on mechanistically relevant variability.

Future Updates to the features table:
1. Adding 'Stepping Direction' with respect to the orientation of the fly
2. Adding inter-leg coordination dependent parameters
3. Adding curvature of trajectory
4. Adding swing amplitude