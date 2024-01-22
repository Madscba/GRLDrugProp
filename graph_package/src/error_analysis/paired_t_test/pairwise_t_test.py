from graph_package.configs.directories import Directories
from graph_package.src.error_analysis.err_utils.err_utils_load import get_saved_pred
from itertools import combinations
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.special as special
import matplotlib.pyplot as plt


def paired_t_test(data1, data2):
    t_statistic, p_value = stats.ttest_rel(data1, data2)

    print("Paired t-test results:")
    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")

    if p_value < 0.05:
        print("The difference between the paired samples is statistically significant.")
        reject_null_hypothesis = True
    else:
        print("There is not enough evidence to reject the null hypothesis.")
        reject_null_hypothesis = False

    return t_statistic, p_value, reject_null_hypothesis


def residual_analysis(
    residual_mse, m1_mse, m2_mse, model_name1, model_name2, plot_residuals
):
    # Check normality using Shapiro-Wilk test
    shapiro_stat, shapiro_p_value = stats.shapiro(residual_mse)
    print(f"Shapiro-Wilk Test - Statistic: {shapiro_stat}, P-Value: {shapiro_p_value}")
    # Check homogeneity of variances using Bartlett's test
    bartlett_stat, bartlett_p_value = stats.bartlett(m1_mse, m2_mse)
    print(f"Bartlett's Test - Statistic: {bartlett_stat}, P-Value: {bartlett_p_value}")
    if plot_residuals:
        # Plot residuals and a Q-Q plot
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(range(1, len(residual_mse) + 1), residual_mse, color="blue")
        plt.axhline(y=0, color="red", linestyle="--")
        plt.title(f"Residuals Plot {model_name1} vs {model_name2}")
        plt.xlabel("Observation")
        plt.ylabel("Residuals")
        plt.subplot(1, 2, 2)
        stats.probplot(residual_mse, plot=plt)
        plt.title(f"Q-Q Plot {model_name1} vs {model_name2}")
        plt.tight_layout()
        plt.show()

    print(f"\nAssumption Tests for {model_name1} & {model_name2}:")
    if shapiro_p_value > 0.05:
        print("Shapiro-Wilk test: Residuals are approximately normally distributed.")
        res_is_normal = True
    else:
        print("Shapiro-Wilk test: Residuals do not follow a normal distribution.")
        res_is_normal = False
    if bartlett_p_value > 0.05:
        print("Bartlett's test: Residuals have homogeneity of variances.")
        res_homoscedastic = True
    else:
        print("Bartlett's test: Residuals do not have homogeneity of variances.")
        res_homoscedastic = False
    return res_is_normal, res_homoscedastic


def perform_manual_paired_two_sided_t_test(residuals_mse):
    # perform paired t test
    mu_hat = np.mean(residuals_mse)
    degrees_of_freedom = len(residuals_mse) - 1
    sigma_hat = np.sqrt(np.sum((mu_hat - residuals_mse) ** 2) / (degrees_of_freedom))
    sigma_err_mean = sigma_hat / np.sqrt(len(residuals_mse))
    t_stat = mu_hat / sigma_err_mean
    alpha = 0.05
    # We are performing a two_sided test
    crit_level = alpha / 2
    # todo get t value lookup table
    t_crit = stats.t.ppf(
        1 - crit_level, degrees_of_freedom
    )  # inverse cdf of t distribution
    p_val = 2 * (stats.t.cdf(-abs(t_stat), degrees_of_freedom))
    if abs(t_stat) > t_crit:
        print("reject null hypothesis: There is a difference in the means")
        print(f"T-statistic {t_stat} and T-crit: {t_crit}\np-val: {p_val}")
        reject_null_hypothesis = True
    else:
        print("accept null hypothesis: There is no difference in the means")
        print(f"T-statistic {t_stat} and T-crit: {t_crit}\np-val: {p_val}")
        reject_null_hypothesis = False
    return t_stat, p_val, reject_null_hypothesis


def check_if_samples_are_paired(df_m1, df_m2):
    assert len(df_m1) == len(
        df_m2
    ), "Samples are not paired, they have different lengths"
    triplet_col = [
        "drug_molecules_left_id",
        "drug_molecules_right_id",
        "context_features_id",
    ]
    assert (
        (df_m1[triplet_col] == df_m2[triplet_col]).all().all()
    ), "Samples are not paired, they have different triplets"


def run_pairwise_two_sided_ttests(
    model_names,
    err_configs,
    alpha: float = 0.05,
    perform_residual_analysis: bool = False,
    plot_residuals: bool = False,
):
    pred_dfs = get_saved_pred(err_configs)

    # prepare pairwise combinations
    model_indices = np.arange(len(model_names))
    combinations_list = list(combinations(model_indices, 2))
    # Bonferroni correction, reject null hypothesis if p-value < alpha / m
    alpha /= len(combinations_list)

    residual_cols = (
        ["res_is_normal", "res_homoscedastic"] if perform_residual_analysis else []
    )

    results_df = pd.DataFrame(
        columns=[
            "model1",
            "model2",
            "t_statistic",
            "p_value",
            "alpha_bonferroni",
            "reject_null_hypothesis",
        ]
        + residual_cols
    )
    for combination in combinations_list:
        m1_idx, m2_idx = combination
        df_m1, df_m2 = pred_dfs[m1_idx], pred_dfs[m2_idx]
        m1_mse = (df_m1["targets"].values - df_m1["predictions"].values) ** 2
        m2_mse = (df_m2["targets"].values - df_m2["predictions"].values) ** 2
        check_if_samples_are_paired(df_m1, df_m2)
        paired_residuals = m1_mse - m2_mse

        (
            t_statistic,
            p_value,
            reject_null_hypothesis,
        ) = perform_manual_paired_two_sided_t_test(paired_residuals)

        # sanity check with scipy
        #t_statistic_, p_value_, reject_null_hypothesis_ = paired_t_test(m1_mse, m2_mse)

        if perform_residual_analysis:
            res_is_normal, res_homoscedastic = residual_analysis(
                paired_residuals,
                m1_mse,
                m2_mse,
                model_names[m1_idx],
                model_names[m2_idx],
                plot_residuals,
            )

        res_values = (
            [res_is_normal, res_homoscedastic] if perform_residual_analysis else []
        )

        results_df.loc[-1] = [
            model_names[m1_idx],
            model_names[m2_idx],
            t_statistic,
            p_value,
            alpha,
            reject_null_hypothesis,
        ] + res_values

    return results_df


if __name__ == "__main__":
    # put prediction files in the folder, where each prediction file is named after the model:
    # ex : rescal_model_pred_dict.pkl

    model_1_config = {
        "task": "reg",
        "target": "zip_mean",
        "day_of_prediction": "22_01_2024",
        "prediction_file_name": "NO_NAME_pred_co3ky6olh8sxpsy24h3uxc1u.pkl",
        "bar_plot_config": {"add_bar_info": True},
    }

    model_2_config = {
        "task": "reg",
        "target": "zip_mean",
        "day_of_prediction": "22_01_2024",
        "prediction_file_name": "NO_NAME_pred_co3ky6olh8sxpsy24h3uxc1u.pkl",
        "bar_plot_config": {"add_bar_info": True},
    }

    err_configs = {
        0: model_1_config,
        1: model_2_config,
    }


    model_names = ["rescal", "deepdds"]
    alpha = 0.05
    perform_residual_analysis = True
    plot_residuals = False

    df_results = run_pairwise_two_sided_ttests(
        model_names,
        err_configs,
        alpha,
        perform_residual_analysis,
        plot_residuals,
    )
