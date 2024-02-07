import numpy as np
import pandas as pd
import pickle
import glob

if __name__ == "__main__":
    #load all file names from a specified folder
    file_name = glob.glob(r"C:\Users\Mads-\Downloads\OneDrive_1_2-2-2024\final_concatenated\*")
    #find files where _10_, _100_, _250_ is in the file name
    file10_name = [f for f in file_name if "_10_" in f]
    file100_name = [f for f in file_name if "_100_" in f]
    file250_name = [f for f in file_name if "_250_" in f]
    #load all file names from a specified folder
    file10 = []
    file100 = []
    file250 = []
    for file in file10_name:
        f = pd.read_pickle(file)
        file10.append(f)
    for file in file100_name:
        f = pd.read_pickle(file)
        file100.append(f)
    for file in file250_name:
        f = pd.read_pickle(file)
        file250.append(f)
    #for each list of files find the set of indexes
    arrays = [pd.DataFrame(file10[i]['batch'][0].numpy()) for i in range(len(file10))]
    common_rows = np.all(np.isin(arrays[0], arrays[1:]), axis=1)
    result_arrays = [array[common_rows] for array in arrays]



# test_df = sorted_df.reset_index()
# model_names = ["RGAT"]
# plt_colors = MODEL_COLORS["rgat"]
# # Set up the subplots
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
#
# #
# #plt_colors_dict = {model_names[i]: col for i, col in enumerate(plt_colors)}
#
# sns.barplot(x=test_df.index, y =test_df['MSE'], data=test_df, ax=axes[0], color=plt_colors)  # Updated this line
# sns.barplot(x=test_df.index, y =test_df['mean_var_cell'], data=test_df, ax=axes[1], color=plt_colors)  # Updated this line
# plt.tight_layout()
# plt.show()
# a = 2


