cx_constant_path=$(python ./environment_config/update_chemicalx.py)
rm -rf $cx_constant_path
parentdir="$(dirname "$cx_constant_path")"
cp ./environment_config/constants.py $parentdir
