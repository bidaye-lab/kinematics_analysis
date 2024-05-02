Analysis of leg kinmatics data

# How to use this repo
For more information on the structure of this repo, 
see this [template repo](https://github.com/bidaye-lab/template_data_pipelines).

## Analysis pipelines
The notebook files `notebooks/*.ipynb` are workflows for the individual steps in the analysis pipeline.

|notebook file|use case|
|---|---|
|[`Gen_DataStructure.ipynb`](notebooks/generate_datastructure/Gen_DataStructure.ipynb)| generate the data structure by combining anipose output & ball velocity & position files; input for ball fitting |
|[`ball_fitting_example.ipynb`](notebooks/ball_fitting_example.ipynb)| fit ball to tarsal tips and predict swing and stance phase per frame and leg|
|[`ball_fitting_batch.ipynb`](notebooks/ball_fitting_batch.ipynb)| run ball fitting and stepcycle predictions in batch mode |

|[`Generate_Features_Table.ipynb`](notebooks/Generate_features_table/Generate_Features_Table.ipynb)| generate stepping features table from the output of ball fitting |

## Documentation
The analysis pipelines contain most information needed to understand how to work with the data.
Some additional information is provided for:
|file|content|
|---|---|
|[data_structure.md](docs/data_structure.md)|Data structure generated with DLC/anipose|
|[3D_visualization.md](docs/3D_visualization.md)|3D visualization using VMD molecular viewer|

## old scripts
These old scripts need to become part of the new code structure.
- `notebooks/old_notebooks/feature_generation/coordinate_transformation.ipynb` incl `utils.py`
- `notebooks/old_notebooks/feature_extraction` (i) BPN, (ii) P9LT, (iii), P9RT
- `notebooks/old_notebooks/generate-dataset` MATLAB code (port to python?)
- `notebooks/old_notebooks/regression_model/model.ipynb`

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

