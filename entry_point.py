import importlib
import sys
from jaxtyping import install_import_hook
from rich import print as rprint


def main():
    # Parse command-line args
    if len(sys.argv) < 2:
        rprint("Usage: python entry_point.py <command> [args...]")
        rprint("Commands:")
        rprint("- train: Run the training script")
        rprint("- infer: Run the inference script")
        rprint("- eval: Run the evaluation script")
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    # Define the mapping of commands to scripts
    command_map = {
        "train": "run_train.py",
        "infer": "run_inference.py",
        "eval": "run_evals.py"
    }

    if command not in command_map:
        print(f"Unknown command: {command}")
        print("Available commands: train, infer, eval")
        sys.exit(1)

    with install_import_hook(["transformer"], "typeguard.typechecked"):
        if command == "train":
            run_train = importlib.import_module("run_train")
            run_train.main(args)
        elif command == "infer":
            run_infer = importlib.import_module("run_inference")
            run_infer.main(args)
        elif command == "eval":
            run_eval = importlib.import_module("run_evals")
            run_eval.main(args)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
