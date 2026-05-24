from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


SUMMARY_FILE = Path("outputs/grid_search_summary.csv")
PLOT_DIR = Path("outputs/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_FILE}")

    df = pd.read_csv(SUMMARY_FILE)

    if df.empty:
        raise ValueError("Summary CSV is empty.")

    # Sort by STOI, then SI-SDR
    df_sorted = df.sort_values(by=["mean_stoi", "mean_si_sdr"], ascending=False)
    df_sorted.to_csv(PLOT_DIR / "sorted_results.csv", index=False)

    print("\nTop 5 results:")
    print(df_sorted.head(5)[["exp_name", "lambda1_spec", "lambda2_smooth", "mean_stoi", "mean_si_sdr"]])

    # -----------------------------
    # Heatmap: STOI
    # -----------------------------
    pivot_stoi = df.pivot(index="lambda2_smooth", columns="lambda1_spec", values="mean_stoi")
    plt.figure(figsize=(8, 6))
    plt.imshow(pivot_stoi.values, aspect="auto")
    plt.xticks(range(len(pivot_stoi.columns)), pivot_stoi.columns)
    plt.yticks(range(len(pivot_stoi.index)), pivot_stoi.index)
    plt.xlabel("lambda1 (spec_w)")
    plt.ylabel("lambda2 (smooth_w)")
    plt.title("Heatmap of Mean STOI")
    plt.colorbar(label="Mean STOI")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "heatmap_stoi.png", dpi=300)
    plt.close()

    # -----------------------------
    # Heatmap: SI-SDR
    # -----------------------------
    pivot_sisdr = df.pivot(index="lambda2_smooth", columns="lambda1_spec", values="mean_si_sdr")
    plt.figure(figsize=(8, 6))
    plt.imshow(pivot_sisdr.values, aspect="auto")
    plt.xticks(range(len(pivot_sisdr.columns)), pivot_sisdr.columns)
    plt.yticks(range(len(pivot_sisdr.index)), pivot_sisdr.index)
    plt.xlabel("lambda1 (spec_w)")
    plt.ylabel("lambda2 (smooth_w)")
    plt.title("Heatmap of Mean SI-SDR")
    plt.colorbar(label="Mean SI-SDR")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "heatmap_sisdr.png", dpi=300)
    plt.close()

    # -----------------------------
    # Pareto-style scatter plot
    # x = SI-SDR
    # y = STOI
    # -----------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(df["mean_si_sdr"], df["mean_stoi"])

    for _, row in df.iterrows():
        plt.annotate(
            f'l1={row["lambda1_spec"]}, l2={row["lambda2_smooth"]}',
            (row["mean_si_sdr"], row["mean_stoi"]),
            fontsize=7,
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.xlabel("Mean SI-SDR")
    plt.ylabel("Mean STOI")
    plt.title("Pareto-style Trade-off Plot")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "pareto_stoi_sisdr.png", dpi=300)
    plt.close()

    print("\nPlots saved in:", PLOT_DIR)


if __name__ == "__main__":
    main()