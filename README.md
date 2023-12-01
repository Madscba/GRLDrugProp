Getting started
============
Follow the steps in this README to run the codebase of this project.

1 Clone repo 
----------------
1) Clone repository to your machine.


2 Create virtual environment
---------------- 

### Conda envirnoment

Prerequisites: C++ Compiler (Instruction for how to install this can be found beneath), 

1) Windows:
    * conda env create --file environment.yml  
    * conda activate grap_pgk_env

2) osx-64 (Intel chip):
    * conda env create --file environment.yml  
    * conda activate grap_pgk_env

3) arm-64 (Apple chip):
    * CONDA_SUBDIR=osx-64 conda env create --file environment.yml 
    * conda activate grap_pgk_env


**PLEASE FIRST CHECK IF NEW PACKAGES ARE AVAILABLE THROUGH CONDA CHANNELS AND ADD THEM TO ```environment.yml```. USE ```pip``` AS LAST RESORT AND ADD TO ```requirements.txt```**

To update environment use ```conda env update --name graph_pkg_env --file environment.yml --prune```.  

### Python venv (for hpc use)
To install env on HPC at DTU use: 
```bash
module load python3/3.9.17 
module load cuda/12.1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_hpc.txt
```
To test it run in same terminal:

```bash
voltash
python graph_package/src/main.py
```
When submitting jobs you have to provide ```module load cuda/12.1``` in your bash script. 

4 Naming convention of runs in W&B
----------------
```bash
{$model}_{$task}_{$target}_{$induction_split}_{any_other_important_change_from_config_you_want}
```

Always use naming of command line args for naming.
 

3 Add pre-commit hooks
----------------
We are using pre-commit to apply basic quality checks and autoformatting on the code.
   
```bash
pre-commit install
pre-commit autoupdate
 ```

On success, you should get a message like: ``pre-commit installed at .git/hooks/pre-commit``.

4 Check that pre-commit work as intended. 
-----------------------------
execute "pre-commit run --all-files" after environment has been setup
```bash
pre-commit run --all-files
 ```


5 Install C++ Compiler via Visual Studio Installer (Prerequisite)
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