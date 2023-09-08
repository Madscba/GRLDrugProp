Copyright © 2022 by Halfspace. All rights reserved

Getting started
============
Follow the steps in this README to run the codebase of this project.


1 Install Poetry
--------------------------
1) Install poetry (https://python-poetry.org/docs/#installation)
   * Linux, macOS, Windows (WSL)
      * "curl -sSL https://install.python-poetry.org | python3 -"
   * Update to latest version: "poetry self update"
   * (Recommended) If you prefer poetry to save your virtual environment configurations in the project folder, then run "poetry config virtualenvs.in-project true"
2) Verify that the command "poetry --version" is recognized.


* Poetry basic commands
  * "poetry install" - Initialize poetry environment from pyproject.toml.
  * "poetry shell" - The easiest way to activate the virtual environment is to create a nested shell.     
  * "exit" - To deactivate the virtual environment and exit this new shell type.
  * "deactivate" - To deactivate the virtual environment without leaving the shell use.
  * "poetry add <package_name>" - Add package to dependencies. 
  * "poetry remove <package_name>" - Remove package from dependencies. 
  * "poetry update" - Update poetry dependencies. 

--------------



2 Clone repo 
----------------
1) Clone repository to your machine.


3 Create virtual environment
---------------- 
1) Create virtual environment using the poetry.lock file (if .lock files does not exist then pyproject.toml will be used).
   * Navigate to project root folder in your terminal 
```bash 
poetry install
 ```
2) (Optional) Modify project packages to suit your needs. **Use poetry for this** and not pip
      * "poetry add <package_name>" 
        * Note that this command replaces the "pip install <>". The command install the package, and adds it to the pyproject.toml file.
      * For documentation on how to use poetry: https://python-poetry.org/docs/basic-usage/

4 Add pre-commit hooks
----------------
We are using pre-commit to apply basic quality checks and autoformatting on the code.
   
```bash
pre-commit install
pre-commit autoupdate
 ```

On success, you should get a message like: ``pre-commit installed at .git/hooks/pre-commit``.

5 Ensure tests can run
----------------------

The repository uses [pytest](https://docs.pytest.org/en/latest/) to run the tests.

In order to check that everything works as expected, run the tests from the root of the repo by first opening the poetry shell with:
```bash
poetry shell
poetry run pytest
 ```


6 Autoformat code with black
----------------------
* poetry run black X  (X can either be a filename to be formatted or a . which formats the whole project)
```bash
poetry shell
poetry run black .
 ```

7 Run ruff for auto-format suggestions.
----------------------
* poetry run ruff X  (X can either be a filename to be formatted or a . which formats the whole project)
* Fix problems automatically: poetry run ruff . --fix option

```bash
poetry shell
poetry run ruff . --fix
 ```


8 Check that pre-commit work as intended. 
-----------------------------
execute "pre-commit run --all-files" after environment has been setup
```bash
pre-commit run --all-files
 ```
Happy Coding!