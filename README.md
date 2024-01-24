Analysis of leg kinmatics data

# How to use this repo
For more information on the structure of this repo, 
see this [template repo](https://github.com/bidaye-lab/template_data_pipelines).

## Analysis pipelines
The script files `scripts/*.py` are workflows for the individual steps in the analysis pipeline.

|script file|use case|
|---|---|
|`ball_fitting_example.py`| fit ball to tarsal tips and predict swing and stance phase per frame and leg|
|`ball_fitting_batch.py`| run ball fitting and stepcycle predictions in batch mode |

## old scripts
These old scripts need to become part of the new code structure.
- `scripts/old_scripts/feature_generation/coordinate_transformation.ipynb` incl `utils.py`
- `scripts/old_scripts/feature_extraction` (i) BPN, (ii) P9LT, (iii), P9RT
- `scripts/old_scripts/generate-dataset` MATLAB code (port to python?)
- `scripts/old_scripts/regression_model/model.ipynb`


## Installation
```
# create conda environment with necessary dependencies
conda env create -n kinematics_analysis -f environment.yml
conda activate kinematics_analysis

# get source code
git clone https://github.com/bidaye-lab/kinematics_analysis

# install code as local local python module
cd kinematics_analysis
pip install -e .
```

