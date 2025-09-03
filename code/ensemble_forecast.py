import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Average multiple prediction files")
    parser.add_argument("inputs", nargs="+", help="Paths to .npy prediction files")
    parser.add_argument("--out", required=True, help="Output .npy path")
    args = parser.parse_args()

    preds = [np.load(p) for p in args.inputs]
    if not preds:
        raise SystemExit("No prediction files provided")
    stacked = np.stack(preds, axis=0)
    mean_pred = stacked.mean(axis=0)
    np.save(args.out, mean_pred)
    print(f"Ensembled {len(preds)} predictions -> {args.out}")


if __name__ == "__main__":
    main()

