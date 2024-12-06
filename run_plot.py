from glob import glob
import argparse
from gpt.utils import plot_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args()
    logfiles = glob(f"{args.exp_name}/*.jsonl")
    plot_name = f"{args.exp_name}/train_plot_{args.exp_name}.html"
    plot_stats(logfiles, plot_name=plot_name, show_individual_runs=False)
    print(f"Plot saved to {plot_name}")


if __name__ == "__main__":
    main()
