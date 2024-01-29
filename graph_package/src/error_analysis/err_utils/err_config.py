
### CONFIG FOR RESIDUAL SCATTER PLOT ANALYSIS IN run_err_analysis
res_model_1_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "26_01_2024",
    "model_name": "DistMult",
    "prediction_file_name": "p3_distmult_pred_tuk4g2fb5sevg66pittq1w7e.pkl",
    "bar_plot_config": {"add_bar_info": True},
}
res_model_2_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "26_01_2024",
    "model_name": "RGAT",
    "prediction_file_name": "p3_rgat_e3fp_4_pred_bpm179tn7wtqt9ov7svwj3lm.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

res_err_configs = {
    0: res_model_1_config,
    1: res_model_2_config,
}


### CONFIG FOR ENTITY LEVEL ANALYSIS IN run_err_analysis
#DISTMULT

entity_dist_1_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "29_01_2024",
    "prediction_file_name": "p3_distmult_pred_tuk4g2fb5sevg66pittq1w7e.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

entity_dist_2_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "29_01_2024",
    "prediction_file_name": "p3_distmult_seed100_pred_k5dkyzic3s6qajftblnwu6ee.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

#DeepDDS
entity_deep_1_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "29_01_2024",
    "prediction_file_name": "p3_deepdds_pred_re0jx39pevo6oluojwl48rcx.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

#GC
entity_gc_1_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "29_01_2024",
    "prediction_file_name": "p3_gc_pred_1704nmtx23m6limxpqirq11i.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

#RGAT
entity_rgat_1_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "29_01_2024",
    "prediction_file_name": "p3_rgat_e3fp_4_pred_bpm179tn7wtqt9ov7svwj3lm.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

entity_rgat_one_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "29_01_2024",
    "prediction_file_name": "p3_rgat__pred_ze3qdbnonql0nmdzk5j03hwh.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

entity_err_configs = {
    0: entity_dist_1_config,
    1: entity_deep_1_config,
    2: entity_gc_1_config,
    3: entity_rgat_1_config,
    4: entity_rgat_one_config,
    # 5: entity_dist_2_config,
}

