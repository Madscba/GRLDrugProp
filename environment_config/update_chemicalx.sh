export cx_constant_path='.venv/lib/python3.9/site-packages/chemicalx/constants.py'
rm -rf $cx_constant_path
parentdir="$(dirname "$cx_constant_path")"
cp ./environment_config/constants.py $parentdir
