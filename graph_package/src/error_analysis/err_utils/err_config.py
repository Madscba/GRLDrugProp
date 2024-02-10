
### CONFIG FOR DistMult vs RGAT p3 gen split
res_model_1_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "26_01_2024",
    "model_name": "DistMult",
    "prediction_file_name": "p3_distmult_pred_tuk4g2fb5sevg66pittq1w7e.pkl",
    "bar_plot_config": {"add_bar_info": True, "y_lim": (0, 165), "model_name": "DistMult"},
}
res_model_2_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "26_01_2024",
    "model_name": "RGAT",
    "prediction_file_name": "p3_rgat_e3fp_4_pred_bpm179tn7wtqt9ov7svwj3lm.pkl",
    "bar_plot_config": {"add_bar_info": True,"y_lim": (0, 165), "model_name": "RGAT"},
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


### CONFIG FOR MODALITY FEWSHOT 10  PAIRWISE TESTS
RGAT_onehot_10config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "30_01_2024",
    "prediction_file_name": "p2_fewshot_drug_10_rgat_k4_onehot_pred_0gddmqu1z77lo0eacdxhu2n5.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

RGAT_onehot_het_10config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "30_01_2024",
    "prediction_file_name": "p2_fewshot_drug_10_rgat_k4_onehot_het_pred_hjysdoypmmumzpd0l9fxcn2r.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

RGAT_3d_het_10config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "30_01_2024",
    "prediction_file_name": "p2_fewshot_drug_10_rgat_k4_3d_het_pred_cci7zhufud4yw4xdear5m8ol.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

RGAT_3d_10config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "30_01_2024",
    "prediction_file_name": "p2_fewshot_drug_10_rgat_k4_3d_pred_krflu438wfcau7k3vhney8oh.pkl",
    "bar_plot_config": {"add_bar_info": True},
}


fewhot_modal_10_err_configs = {
    0: RGAT_onehot_10config,
    1: RGAT_onehot_het_10config,
    2: RGAT_3d_het_10config,
    3: RGAT_3d_10config,
}


### CONFIG FOR MODALITY FEWSHOT 100 PAIRWISE TESTS
RGAT_onehot_100config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "30_01_2024",
    "prediction_file_name": "p2_fewshot_drug_100_rgat_k4_onehot_pred_hj88n9jq6ggb7fkfcr1ntwmu.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

RGAT_onehot_het_100config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "30_01_2024",
    "prediction_file_name": "p2_fewshot_drug_100_rgat_k4_onehot_het_pred_6ns1xo4q21rml6em2u3srwfr.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

RGAT_3d_het_100config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "30_01_2024",
    "prediction_file_name": "p2_fewshot_drug_100_rgat_k4_3d_het_pred_zpyxem9yayxedy3cjtfsj4xd.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

RGAT_3d_100config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "30_01_2024",
    "prediction_file_name": "p2_fewshot_drug_100_rgat_k4_3d_pred_w9ivc7v768xqwczo3x47n6yb.pkl",
    "bar_plot_config": {"add_bar_info": True},
}


fewhot_modal_100_err_configs = {
    0: RGAT_onehot_100config,
    1: RGAT_onehot_het_100config,
    2: RGAT_3d_het_100config,
    3: RGAT_3d_100config,
}

### CONFIG FOR MODALITY FEWSHOT 250 PAIRWISE TESTS
RGAT_onehot_250config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "30_01_2024",
    "prediction_file_name": "p2_fewshot_drug_250_rgat_k4_onehot_pred_kdx6hkq7rlygh9ysfnxekg60.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

RGAT_onehot_het_250config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "30_01_2024",
    "prediction_file_name": "p2_fewshot_drug_250_rgat_k4_onehot_het_pred_g17jryi97ns6pqpyltwmdz7a.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

RGAT_3d_het_250config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "30_01_2024",
    "prediction_file_name": "p2_fewshot_drug_250_rgat_k4_3d_het_pred_6rajivuairjvohbct1pknxes.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

RGAT_3d_250config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "30_01_2024",
    "prediction_file_name": "p2_fewshot_drug_250_rgat_k4_3d_pred_2gkex6h5d54xhxgoe3oh38rr.pkl",
    "bar_plot_config": {"add_bar_info": True},
}


fewhot_modal_250_err_configs = {
    0: RGAT_onehot_250config,
    1: RGAT_onehot_het_250config,
    2: RGAT_3d_het_250config,
    3: RGAT_3d_250config,
}



#PAIRWISE FEW-SHOT DRUG 10 (RGAT VS DeepDDS)
RGAT_3d_10config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "31_01_2024",
    "prediction_file_name": "p3_fewshot_drug_rgat_10_pred_i5rhvjhx8znsjdtzu3dkqlsw.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

DeepDDS_10config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "31_01_2024",
    "prediction_file_name": "p3_fewshot_drug_deepdds_10_pred_uanjlmaqglf2c2m62z2n6314.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

p3_few_shot_drug_10_err_configs = {
    0: RGAT_3d_10config,
    1: DeepDDS_10config,
}

#PAIRWISE FEW-SHOT DRUG 100 (RGAT VS DISTMULT)

RGAT_3d_100config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "31_01_2024",
    "prediction_file_name": "p3_fewshot_drug_rgat_100_pred_48uavybzdt2jg4t8r2iw4ft2.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

DistMult_100config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "31_01_2024",
    "prediction_file_name": "p3_fewshot_drug_distmult_100_pred_qvf7k2wwctdlpxsxreni4p4u.pkl",
    "bar_plot_config": {"add_bar_info": True},
}


p3_few_shot_drug_100_err_configs = {
    0: RGAT_3d_100config,
    1: DistMult_100config,
}

#PAIRWISE FEW-SHOT DRUG 250 (RGAT VS DISTMULT)

RGAT_3d_250config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "31_01_2024",
    "prediction_file_name": "p3_fewshot_drug_rgat_250_pred_la2oofygeba4d02jjgx0fy2y.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

DistMult_250config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "31_01_2024",
    "prediction_file_name": "p3_fewshot_drug_distmult_250_pred_keabqukhhzaepp1ibzw7nfh0.pkl",
    "bar_plot_config": {"add_bar_info": True},
}


p3_few_shot_drug_250_err_configs = {
    0: RGAT_3d_250config,
    1: DistMult_250config,
}


#PAIRWISE FEW-SHOT CELL 100 (RGAT VS DISTMULT)

RGAT_3d_cell_100config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "01_02_2024",
    "prediction_file_name": "p3_fewshot_cell_rgat_100_pred_8d3tqio2gspi8db3pd0vfj2y.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

DistMult_cell_100config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "01_02_2024",
    "prediction_file_name": "p3_fewshot_cell_distmult_100_pred_la0pojgwreumrirqp0gdz5c5.pkl",
    "bar_plot_config": {"add_bar_info": True},
}


p3_few_shot_cell_100_err_configs = {
    0: RGAT_3d_cell_100config,
    1: DistMult_cell_100config,
}

#PAIRWISE FEW-SHOT CELL 250 (RGAT VS DISTMULT)

RGAT_3d_cell_250config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "01_02_2024",
    "prediction_file_name": "p3_fewshot_cell_rgat_250_pred_4kxatm2zogwhfq58jv0mcp4n.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

DistMult_cell_250config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "01_02_2024",
    "prediction_file_name": "p3_fewshot_cell_distmult_250_pred_1pl75m38gg34wv1ede7yy6in.pkl",
    "bar_plot_config": {"add_bar_info": True},
}


p3_few_shot_cell_250_err_configs = {
    0: RGAT_3d_cell_250config,
    1: DistMult_cell_250config,
}



#P3 intra model variablility
RGAT_3d_general_config = {
    "task": "reg",
    "target": "zip_mean",
    "model_name": "RGAT-1",
    "day_of_prediction": "28_01_2024",
    "prediction_file_name": "p3_rgat_e3fp_4_pred_bpm179tn7wtqt9ov7svwj3lm.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

RGAT_3d_general2_config = {
    "task": "reg",
    "target": "zip_mean",
    "model_name": "RGAT-2",
    "day_of_prediction": "28_01_2024",
    "prediction_file_name": "p3_rgat_e3fp_4_pred_jkaj2vbfn3eitubs3lh4w7fx.pkl",
    "bar_plot_config": {"add_bar_info": True},
}


p3_general_intra_model_var_err_configs = {
    0: RGAT_3d_general_config,
    1: RGAT_3d_general2_config,
}


## p3 RGAT one-hot vs E3FP fewshot drug 10

rgat_drug_one_hot_10_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "31_01_2024",
    "model_name": "RGAT-one-hot",
    "prediction_file_name": "p3_fewshot_drug_rgat_onehot_10_pred_0qvgxtg4zi8vwsjt0gscg88d.pkl",
    "bar_plot_config": {"add_bar_info": True},
}
rgat_drug_e3fp_10_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "31_01_2024",
    "model_name": "RGAT-E3FP",
    "prediction_file_name": "p3_fewshot_drug_rgat_10_pred_i5rhvjhx8znsjdtzu3dkqlsw.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

p3_rgat_drug_10 = {
    0: rgat_drug_one_hot_10_config,
    1: rgat_drug_e3fp_10_config,
}

## p3 RGAT one-hot vs E3FP fewshot drug 100


rgat_drug_one_hot_100_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "31_01_2024",
    "model_name": "RGAT-one-hot",
    "prediction_file_name": "p3_fewshot_drug_rgat_onehot_100_pred_86ie40p8y0hruaxf2n663c5t.pkl",
    "bar_plot_config": {"add_bar_info": True},
}
rgat_drug_e3fp_100_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "31_01_2024",
    "model_name": "RGAT-E3FP",
    "prediction_file_name": "p3_fewshot_drug_rgat_100_pred_48uavybzdt2jg4t8r2iw4ft2.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

p3_rgat_drug_100 = {
    0: rgat_drug_one_hot_100_config,
    1: rgat_drug_e3fp_100_config,
}

## p3 RGAT one-hot vs E3FP fewshot drug 250


rgat_drug_one_hot_250_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "31_01_2024",
    "model_name": "RGAT-one-hot",
    "prediction_file_name": "p3_fewshot_drug_rgat_onehot_250_pred_4fgqa64i2j85khklm6a5h0sk.pkl",
    "bar_plot_config": {"add_bar_info": True},
}
rgat_drug_e3fp_250_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "31_01_2024",
    "model_name": "RGAT-E3FP",
    "prediction_file_name": "p3_fewshot_drug_rgat_250_pred_la2oofygeba4d02jjgx0fy2y.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

p3_rgat_drug_250 = {
    0: rgat_drug_one_hot_250_config,
    1: rgat_drug_e3fp_250_config,
}


## p3 RGAT one-hot vs E3FP fewshot cell 10

rgat_cell_one_hot_10_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "01_02_2024",
    "model_name": "RGAT-one-hot",
    "prediction_file_name": "p3_fewshot_cell_rgat_onehot_10_pred_pikqeywap5cyudozey63ehtp.pkl",
    "bar_plot_config": {"add_bar_info": True},
}
rgat_cell_e3fp_10_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "01_02_2024",
    "model_name": "RGAT-E3FP",
    "prediction_file_name": "p3_fewshot_cell_rgat_10_pred_sr7rusoo9nsgywpv1q8t30mb.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

p3_rgat_cell_10 = {
    0: rgat_cell_one_hot_10_config,
    1: rgat_cell_e3fp_10_config,
}

## p3 RGAT one-hot vs E3FP fewshot drug 100

rgat_cell_one_hot_100_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "01_02_2024",
    "model_name": "RGAT-one-hot",
    "prediction_file_name": "p3_fewshot_cell_rgat_onehot_100_pred_rrh7lnwxlj2smyyyva61bdnx.pkl",
    "bar_plot_config": {"add_bar_info": True},
}
rgat_cell_e3fp_100_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "01_02_2024",
    "model_name": "RGAT-E3FP",
    "prediction_file_name": "p3_fewshot_cell_rgat_100_pred_8d3tqio2gspi8db3pd0vfj2y.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

p3_rgat_cell_100 = {
    0: rgat_cell_one_hot_100_config,
    1: rgat_cell_e3fp_100_config,
}

## p3 RGAT one-hot vs E3FP fewshot drug 250

rgat_cell_one_hot_250_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "01_02_2024",
    "model_name": "RGAT-one-hot",
    "prediction_file_name": "p3_fewshot_cell_rgat_onehot_250_pred_g0xmz4q4afpsmuzsltezeiuo.pkl",
    "bar_plot_config": {"add_bar_info": True},
}
rgat_cell_e3fp_250_config = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "01_02_2024",
    "model_name": "RGAT-E3FP",
    "prediction_file_name": "p3_fewshot_cell_rgat_250_pred_4kxatm2zogwhfq58jv0mcp4n.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

p3_rgat_cell_250 = {
    0: rgat_cell_one_hot_250_config,
    1: rgat_cell_e3fp_250_config,
}

## p3 E3FP 388 vs shuffled version

rgat_e3fp_1 = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "09_02_2024",
    "model_name": "RGAT-one-hot",
    "prediction_file_name": "p3_rgat_e3fp_388_1_pred_22xjkpo4jdnmdzhxi6bu97gz.pkl",
    "bar_plot_config": {"add_bar_info": True},
}
rgat_e3fp_shuffle = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "09_02_2024",
    "model_name": "RGAT-E3FP",
    "prediction_file_name": "p3_rgat_e3fp_388_1_shuffle_pred_a5sznlazs53ok3zzju41kd6o.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

p3_e3fp_shuffle = {
    0: entity_rgat_one_config,  # one-hot 388
    1: rgat_e3fp_1,
    2: rgat_e3fp_shuffle,
}

# p2 modality vs one-hot, general

p2_rgat_one_hot_1 = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "09_02_2024",
    "model_name": "RGAT-one-hot",
    "prediction_file_name": "p2_3_rgac_pred_ywc4cd85dcdu770tqo2mxo07.pkl",
    "bar_plot_config": {"add_bar_info": True},
}
p2_rgat_modality = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "09_02_2024",
    "model_name": "RGAT-E3FP",
    "prediction_file_name": "p2_3_rgac_modalities_only_pred_82y6niu700dfqlaejr1clsmb.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

p2_modal_onehot = {
    0: p2_rgat_one_hot_1,
    1: p2_rgat_modality,
}

#p2 few-shot modality exp. one-hot  vs het (without-one-hot)


p2_rgat_het_10_no_one_hot = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "10_02_2024",
    "model_name": "RGAT",
    "prediction_file_name": "p2_fewshot_drug_10_rgat_k4_het2_pred_5168hep0rwjraldfw151c35s.pkl",
    "bar_plot_config": {"add_bar_info": True},
}
p2_few_shot_10_het_no_one_hot = {
    0: RGAT_onehot_10config,
    1: p2_rgat_het_10_no_one_hot,
}

p2_rgat_het_100_no_one_hot = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "10_02_2024",
    "model_name": "RGAT",
    "prediction_file_name": "p2_fewshot_drug_100_rgat_k4_het2_pred_1bq4j7svwx0n4km5kn8tw73d.pkl",
    "bar_plot_config": {"add_bar_info": True},
}

p2_few_shot_100_het_no_one_hot = {
    0: RGAT_onehot_100config,
    1: p2_rgat_het_100_no_one_hot,
}

p2_rgat_het_250_no_one_hot = {
    "task": "reg",
    "target": "zip_mean",
    "day_of_prediction": "10_02_2024",
    "model_name": "RGAT",
    "prediction_file_name": "p2_fewshot_drug_250_rgat_k4_het2_pred_u62812guqrnwu5ksjy3mfwes.pkl",
    "bar_plot_config": {"add_bar_info": True},
}
p2_few_shot_250_het_no_one_hot = {
    0: RGAT_onehot_250config,
    1: p2_rgat_het_250_no_one_hot,
}



