import pandas as pd
import torch


def format_and_return_as_dataframes(pred_dict):
    """Error diagnostics function that unpacks the prediction dictionary that holds batch triplets, preductions and targets and was saved under trainer.Test() and returns it as a dataframe"""
    if len(pred_dict["batch"]) != 2:
        fold_triplets = torch.vstack(pred_dict["batch"][::2]).numpy()
        fold_predictions = torch.hstack(pred_dict["predictions"]).numpy()
        fold_target = torch.hstack(pred_dict["batch"][1::2]).numpy()
        dataframes = [
            pd.DataFrame(ent)
            for ent in [
                fold_triplets,
                fold_predictions,
                fold_target,
            ]
        ]
    else:
        dataframes = [
            pd.DataFrame(ent)
            for ent in [
                pred_dict["batch"][0],
                pred_dict["predictions"],
                pred_dict["batch"][1],
            ]
        ]
    df = pd.concat(dataframes, axis=1)
    df.columns = [
        "drug_molecules_left_id",
        "drug_molecules_right_id",
        "context_features_id",
        "predictions",
        "targets",
    ]
    return df
