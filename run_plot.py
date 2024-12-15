from glob import glob
import argparse
from gpt.utils import plot_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args()
    logfile = glob(f"results/{args.exp_name}/*.jsonl")[0]
    plot_name = f"{args.exp_name}/train_plot_{args.exp_name}.html"
    plot_stats(logfile, plot_name=plot_name)
    print(f"Plot saved to {plot_name}")


if __name__ == "__main__":
    main()
