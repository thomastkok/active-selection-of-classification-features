import argparse
import glob
import os
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import ROOT_DIR

# Note: This code was used to create learning curves, for the experiments.
# The code does not work exactly as intended for each combination, and
# needs to be changed depending on the expected outcome. For that reason,
# a notebook might be more practical for creating learning curves.


def read_from_conf(path, metric):
    results_list = []
    for root, dirs, files in os.walk(path):
        if dirs == []:
            for fold in files:
                results = pd.read_csv(os.path.join(root, fold), index_col=0).fillna(0)
                results_list.append(results[metric])
    results_df = pd.DataFrame(results_list).T
    return results_df


def evaluate_run(path, metric):
    df = read_from_conf(path, metric)
    fig, ax = plt.subplots()
    plot_curve(df, metric, ax=ax)
    plt.show()


def evaluate(
    path, metric, method_name="N/A", random=True, compare=None, var=None, save=False
):
    fig, axs = plt.subplots(
        3,
        1 if method_name == "Prob" else 2,
        figsize=(5, 10) if "Prob" in method_name else (10, 10),
        sharey=True,
    )
    if "synth" not in path and "case-study" not in path and "schiz" not in path:
        for i, a in zip(["BreastCancer", "HeartDisease", "Wine"], axs):
            if method_name != "Prob":
                for j, ax in zip(["LogReg", "SVM"], a):
                    results_df = read_from_conf(os.path.join(path, j, i), metric)
                    if compare:
                        comp = [
                            read_from_conf(os.path.join(x, j, i), metric)
                            for x in compare
                        ]
                    plot_curve(
                        results_df,
                        metric,
                        ax=ax,
                        random=random,
                        var=var,
                        comp=comp if compare else None,
                        clf=j,
                        ds=i,
                    )

                    if j == "LogReg":
                        handles, _ = ax.get_legend_handles_labels()
                        fig.legend(
                            handles,
                            [
                                "Random",
                                "ACFA-AVID",
                                "EM-AVID",
                                "APCS",
                                "ACFA-Prob",
                                "EM-Prob",
                            ],
                            loc="right",
                            bbox_to_anchor=(1, 0.15),
                            fancybox=True,
                        )
            else:
                j = "LogReg"
                results_df = read_from_conf(os.path.join(path, j, i), metric)
                if compare:
                    comp = [
                        read_from_conf(os.path.join(x, j, i), metric) for x in compare
                    ]
                plot_curve(
                    results_df,
                    metric,
                    ax=a,
                    random=random,
                    var=var,
                    comp=comp if compare else None,
                    clf=j,
                    ds=i,
                )

                handles, _ = a.get_legend_handles_labels()
                fig.legend(
                    handles,
                    [method_name, "Random"],
                    ["$n = 7$", "BIC", "Random"],
                    loc="right",
                    bbox_to_anchor=(1, 0.15),
                    fancybox=True,
                )

        if method_name != "Prob":
            plt.setp(axs[-1, :], xlabel="Queried instances")
        else:
            plt.setp(axs[-1], xlabel="Queried instances")

        if method_name == "Prob":
            for ax, row in zip(axs[:], ["BreastCancer", "HeartDisease", "Wine"]):
                ax.set_ylabel(row)
        else:
            for ax, col in zip(axs[0], ["LogReg", "SVM"]):
                ax.set_title(col)

            for ax, row in zip(axs[:, 0], ["BreastCancer", "HeartDisease", "Wine"]):
                ax.set_ylabel(row)
        fig.tight_layout()
        plt.show()
    elif "synth" in path:
        for i, a in zip(["Direct", "Indirect", "None"], axs):
            if method_name != "Prob":
                for j, ax in zip(["LogReg", "SVM"], a):
                    results_df = read_from_conf(os.path.join(path, j, i), metric)
                    if compare:
                        comp = [
                            read_from_conf(os.path.join(x, j, i), metric)
                            for x in compare
                        ]
                    plot_curve(
                        results_df,
                        metric,
                        ax=ax,
                        random=random,
                        var=var,
                        comp=comp if compare else None,
                        clf=j,
                        ds=i,
                    )
            else:
                j = "LogReg"
                results_df = read_from_conf(os.path.join(path, j, i), metric)
                if compare:
                    comp = [
                        read_from_conf(os.path.join(x, j, i), metric) for x in compare
                    ]
                plot_curve(
                    results_df,
                    metric,
                    ax=a,
                    random=random,
                    var=var,
                    comp=comp if compare else None,
                    clf=j,
                    ds=i,
                )

                handles, _ = a.get_legend_handles_labels()
                fig.legend(
                    handles,
                    [method_name, "Random"],
                    loc="right",
                    bbox_to_anchor=(1, 0.15),
                    fancybox=True,
                )
        if method_name != "Prob":
            plt.setp(axs[-1, :], xlabel="Queried instances")
        else:
            plt.setp(axs[-1], xlabel="Queried instances")

        if method_name == "Prob":
            for ax, row in zip(axs[:], ["Direct", "Indirect", "None"]):
                ax.set_ylabel(row)
        else:
            for ax, col in zip(axs[0], ["LogReg", "SVM"]):
                ax.set_title(col)

            for ax, row in zip(axs[:, 0], ["Direct", "Indirect", "None"]):
                ax.set_ylabel(row)

        fig.tight_layout()
        plt.show()
    elif "case-study" in path or "schiz" in path:
        for i, a in zip(["iq", "no-iq"], axs):
            for j, ax in zip(["LogReg", "SVM"], a):
                results_df = read_from_conf(os.path.join(path, j, i), metric)
                if compare:
                    comp = [
                        read_from_conf(os.path.join(x, j, i), metric) for x in compare
                    ]
                plot_curve(
                    results_df,
                    metric,
                    ax=ax,
                    random=random,
                    var=var,
                    comp=comp if compare else None,
                    clf=j,
                    ds=i,
                )

                if j == "LogReg":
                    handles, _ = ax.get_legend_handles_labels()
                    fig.legend(
                        handles,
                        ["ACFA-AVID", "EM-AVID", "Random", "ACFA-Prob", "EM-Prob"],
                        loc="right",
                        bbox_to_anchor=(1, 0.15),
                        fancybox=True,
                    )
        plt.setp(axs[-1], xlabel="Queried instances")
        for ax, col in zip(axs[0], ["LogReg", "SVM"]):
            ax.set_title(col)

        for ax, row in zip(axs[:, 0], ["Instances with IQ score", "All instances"]):
            ax.set_ylabel(row)
        fig.tight_layout()
        plt.show()


def plot_curve(
    data, metric, clf=None, ds=None, ax=None, random=None, var=None, comp=None
):
    if not data.empty:
        mean = data.mean(axis=1)
        ci = data.quantile([0.1, 0.9], axis=1).T
        neg = mean - ci[0.1]
        pos = ci[0.9] - mean

    if not data.empty:
        step_size = len(mean) // 6
        mean.plot(
            ax=ax,
            yerr=[[neg, pos]],
            errorevery=(0, step_size),
            elinewidth=1,
            capsize=2,
        )
    if random:
        random_path = (
            os.path.join(
                ROOT_DIR, "experiments", "synth", "Baseline", "Random", clf, ds
            )
            if ds in ["Direct", "Indirect", "None"]
            else os.path.join(ROOT_DIR, "experiments", "Baseline", "Random", clf, ds)
        )
        random_data = read_from_conf(random_path, metric)
        ci = random_data.quantile([0.1, 0.9], axis=1).T
        neg = random_data.mean(axis=1) - ci[0.1]
        pos = ci[0.9] - random_data.mean(axis=1)
        random_data.mean(axis=1).plot(
            ax=ax, yerr=[[neg, pos]], errorevery=step_size, elinewidth=1, capsize=2,
        ),
    if comp:
        for i, c in enumerate(comp):
            if not c.empty:
                c_mean = c.mean(axis=1)
                c_ci = c.quantile([0.1, 0.9], axis=1).T
                c_neg = c_mean - c_ci[0.1]
                c_pos = c_ci[0.9] - c_mean
                step_size = len(c_mean) // 6
                c_mean.plot(
                    ax=ax,
                    yerr=[[c_neg, c_pos]],
                    errorevery=((i + 1) * step_size // 6, step_size),
                    elinewidth=1,
                    capsize=2,
                )

    ax.set_ylim(0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Evaluate an Active Classification Feature \
    Selection experiment."
    )
    parser.add_argument("experiment", help="The experiment's config file or folder.")
    parser.add_argument("-a", "--approach", help="Approach name.")
    parser.add_argument("-m", "--metric", help="Evaluation metric.")
    parser.add_argument("-v", "--variance", help="Variance visualization.")
    parser.add_argument(
        "--random", action="store_false", help="Remove the random sampled baseline.",
    )
    parser.add_argument(
        "--compare", nargs="+", help="Add more experiments to the plot."
    )
    parser.add_argument("-s", "--save", help="Save the figure.", action="store_true")
    args = parser.parse_args()

    sns.set()

    metric = args.metric if args.metric else "f1"
    approach = args.approach
    if "case-study" not in os.path.dirname(args.experiment):
        evaluate(
            os.path.dirname(args.experiment),
            metric,
            method_name=args.approach,
            random=args.random,
            compare=args.compare,
            var=args.variance,
            save=args.save,
        )
    else:
        evaluate_run(os.path.dirname(args.experiment), metric)