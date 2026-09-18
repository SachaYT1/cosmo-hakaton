"""Единая точка входа инференса, требуемая постановкой кейса.

Запуск:
    python inference.py --data-dir /path/to/test --output /path/to/submission.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fireapp.af.infer import run_af_inference
from fireapp.bs.infer import run_bs_inference
from fireapp.common.config import load_config
from fireapp.pipeline.build_submission import build_submission


def main(data_dir: str, output: str) -> None:
    af_cfg = load_config("configs/af.yaml")
    bs_cfg = load_config("configs/bs.yaml")

    af_preds = run_af_inference(Path(data_dir) / "af", af_cfg)
    bs_preds = run_bs_inference(Path(data_dir) / "bs", bs_cfg)

    build_submission(
        af_preds=af_preds,
        bs_preds=bs_preds,
        sample_submission_path=Path(data_dir) / "sample_submission.csv",
        output_path=output,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.data_dir, args.output)
