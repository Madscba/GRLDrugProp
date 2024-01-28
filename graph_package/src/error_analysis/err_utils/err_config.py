
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
entity_model_1_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "26_01_2024",
    "prediction_file_name": "p3_distmult_pred_tuk4g2fb5sevg66pittq1w7e.pkl",
    "bar_plot_config": {"add_bar_info": True},
}
entity_model_2_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "26_01_2024",
    "prediction_file_name": "p3_rgat_e3fp_4_pred_bpm179tn7wtqt9ov7svwj3lm.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

entity_err_configs = {
    0: entity_model_1_config,
    1: entity_model_2_config,
}

