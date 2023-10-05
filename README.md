Getting started
============
Follow the steps in this README to run the codebase of this project.





1 Clone repo 
----------------
1) Clone repository to your machine.


2 Create virtual environment
---------------- 
Prerequisites: Python 3.7.16, C++ Compiler (Instruction for how to install this can be found beneath), 
1) Windows:
   2) Windows environment setup:
    * conda create -n <myenv> python=3.7.16
    * conda activate <myenv>
    * conda install pytorch==1.10.0 -c pytorch
    * conda install pytorch-scatter -c pyg
    * conda install pytorch-cluster -c pyg
    * conda install torchdrug -c milagraph
    * pip install torchdrug==0.1.3.post1


3 Add pre-commit hooks
----------------
We are using pre-commit to apply basic quality checks and autoformatting on the code.
   
```bash
pre-commit install
pre-commit autoupdate
 ```

On success, you should get a message like: ``pre-commit installed at .git/hooks/pre-commit``.

4 Ensure tests can run
----------------------

The repository uses [pytest](https://docs.pytest.org/en/latest/) to run the tests.

In order to check that everything works as expected, run the tests from the root of the repo by first opening the poetry shell with:
```bash
poetry shell
poetry run pytest
 ```


5 Autoformat code with black
----------------------
* poetry run black X  (X can either be a filename to be formatted or a . which formats the whole project)
```bash
poetry shell
poetry run black .
 ```

6 Run ruff for auto-format suggestions.
----------------------
* poetry run ruff X  (X can either be a filename to be formatted or a . which formats the whole project)
* Fix problems automatically: poetry run ruff . --fix option

```bash
poetry shell
poetry run ruff . --fix
 ```


7 Check that pre-commit work as intended. 
-----------------------------
execute "pre-commit run --all-files" after environment has been setup
```bash
pre-commit run --all-files
 ```


8 Install C++ Compiler via Visual Studio Installer (Prerequisite)
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