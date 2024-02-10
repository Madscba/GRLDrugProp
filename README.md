Getting started
============
Follow the steps in this README to run the codebase of this project.

1 Clone repo 
----------------
1) Clone repository to your machine.


2 Create virtual environment
---------------- 

### Conda environment

Prerequisites: C++ Compiler (Instruction for how to install this can be found beneath), 

1) Windows:
    * conda env create --file environment.yml  
    * conda activate graph_pkg_env

2) osx-64 (Intel chip):
    * conda env create --file environment.yml  
    * conda activate graph_pkg_env

3) arm-64 (Apple chip):
    * CONDA_SUBDIR=osx-64 conda env create --file environment.yml 
    * conda activate grap_pgk_env


### Python venv (for hpc use)
To install env on HPC at DTU use: 
```bash
module load python3/3.9.17 
module load cuda/12.1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_hpc_base.txt
pip install -r requirements_hpc_torch.txt
```
To test it run in same terminal:

```bash
voltash
python graph_package/src/main.py
```

When submitting jobs you have to provide 

```bash 
module load cuda/12.1 
module load gcc/13.1.0-binutils-2.40
``` 

in your bash script. 

3 Update environment with new packages
----------------
Please check if new packages are available through conda channels and add them to ```environment.yml```. Use ```pip``` as last resort and add to ```requirements.txt```.
To update environment use ```conda env update --name graph_pkg_env --file environment.yml --prune```.  


4 Add pre-commit hooks (not required to run code)
----------------
We are using pre-commit to apply basic quality checks and autoformatting on the code.
   
```bash
pre-commit install
pre-commit autoupdate
 ```

On success, you should get a message like: ``pre-commit installed at .git/hooks/pre-commit``.

5 Check that pre-commit work as intended. 
-----------------------------
execute "pre-commit run --all-files" after environment has been setup
```bash
pre-commit run --all-files
 ```


6 Install C++ Compiler via Visual Studio Installer (Prerequisite)
--------------------
Install C++ compiler:
1. Go to VS Installer
2. Click on modify
3. Select Desktop development with C++.

Select subcategories:
* MSVC v143 - VS 2022 C++x64/x86 build tools
* C++ ATL for latest v143 build tools
* C++ profiling tools
* C++ Cmake tools for Windows
* Windows 11 SDK (10.0.22621.0)
* vcpkg package manager

Open "Developer Command Prompt for VS2022" and type "where cl".

Add path the path of the cl.exe to environment variable "PATH"
(ex of path: C:\VS2022\VC\Tools\MSVC\14.10.25017\bin\HostX64\x64).


7 Experimental naming convention of runs to be saved on the Weight & Biases ML platform
----------------
```bash
{$model}_{$task}_{$target}_{Optional: any_other_important_change_from_config_you_want}
```
Always use naming of command line args for naming.

8 Data pipeline  
----------------

Manual download of gene-expression data from DepMap:
1. Download the file OmicsCNGene.csv gene expressions from https://depmap.org/portal/download/all/ 
2. Store it in ~/data/features/cell_line_features/raw

Go to graph_package/src/etl/run.py and run the script to preprocess the data. This will generate all the datasets and features needed.
This file downloads the DrugComb data via the API and preprocesses it. Drug features are generated using RDKit and E3FP. 

9 Run the code
----------------
To run the code, execute the following command in the terminal:
```python graph_package/src/main.py```

To configure the run, use the command line arguments. For example:

``` python graph_package/src/main.py model="gnn" +layer="gc" +task="reg" +prediction_head="mlp" ++trainer.max_epochs=100 ++dataset.target="zip_mean""```



