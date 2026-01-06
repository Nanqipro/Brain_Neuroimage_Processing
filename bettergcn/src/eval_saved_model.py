import argparse
import datetime
import json
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from model import ImprovedGCN
from process import compute_correlation_matrix, create_pyg_dataset, enhance_balanced_dataset, load_data, oversample_data
from train import evaluate_model


def set_seed(seed: int) -> int:
    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def get_device(gpu_id: int) -> torch.device:
    if torch.cuda.is_available():
        if int(gpu_id) >= 0:
            device = torch.device(f"cuda:{int(gpu_id)}")
            torch.cuda.set_device(device)
            return device
        return torch.device("cuda")
    return torch.device("cpu")


def apply_balance_strategy(
    features: np.ndarray,
    labels: np.ndarray,
    strategy: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    strategy = str(strategy or "none").lower()
    if strategy == "none":
        return features, labels
    if strategy == "smote":
        return oversample_data(features, labels, ramdom_state=seed, method="smote")
    if strategy == "adasyn":
        return enhance_balanced_dataset(features, labels, methods=["adasyn"], random_state=seed)
    if strategy == "basic":
        return enhance_balanced_dataset(features, labels, methods=["combined", "timeseries"], random_state=seed)
    if strategy == "gan":
        return enhance_balanced_dataset(features, labels, methods=["random_under", "gan"], random_state=seed)
    if strategy == "vae":
        return enhance_balanced_dataset(features, labels, methods=["random_under", "vae"], random_state=seed)
    if strategy == "comprehensive":
        return enhance_balanced_dataset(features, labels, methods=["combined", "timeseries", "gan", "vae"], random_state=seed)
    raise ValueError(f"不支持的 balance_strategy: {strategy}")


def build_val_loader(
    *,
    data_file: str,
    min_samples: int,
    effect_size_file: str,
    effect_threshold: float,
    effect_filter_mode: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    balance_strategy: str,
    batch_size: int,
) -> Tuple[DataLoader, List[str], int]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError(f"非法划分比例: train_ratio={train_ratio}, val_ratio={val_ratio}")

    features, labels, _, class_names = load_data(
        data_file,
        min_samples=min_samples,
        effect_size_path=effect_size_file,
        effect_threshold=effect_threshold,
        effect_filter_mode=effect_filter_mode,
    )

    features_resampled, labels_resampled = apply_balance_strategy(
        features=features,
        labels=labels,
        strategy=balance_strategy,
        seed=seed,
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        features_resampled,
        labels_resampled,
        test_size=(1 - float(train_ratio)),
        random_state=int(seed),
        stratify=labels_resampled,
    )

    val_size = float(val_ratio) / (1 - float(train_ratio))
    X_val, _, y_val, _ = train_test_split(
        X_temp,
        y_temp,
        test_size=(1 - float(val_size)),
        random_state=int(seed),
        stratify=y_temp,
    )

    correlation_matrix = compute_correlation_matrix(X_train)
    val_data_list = create_pyg_dataset(X_val, y_val, correlation_matrix)
    val_loader = DataLoader(val_data_list, batch_size=int(batch_size), shuffle=False)
    num_classes = int(len(np.unique(labels_resampled)))
    return val_loader, class_names, num_classes


def ensure_output_dir(output_dir: Optional[str], model_path: str) -> str:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(str(model_path)))[0]
    out = os.path.join("..", "eval_results", f"{base}_{ts}")
    os.makedirs(out, exist_ok=True)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True, type=str)
    parser.add_argument("--position_file", default="", type=str)
    parser.add_argument("--min_samples", default=50, type=int)
    parser.add_argument("--effect_size_file", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--train_ratio", default=0.6, type=float)
    parser.add_argument("--val_ratio", default=0.2, type=float)
    parser.add_argument("--gpu_id", default=-1, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--balance_strategy", default="none", type=str)

    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--output_dir", default=None, type=str)

    parser.add_argument("--eval_modes", default="three", choices=["three", "single"])
    parser.add_argument("--effect_threshold", default=0.5, type=float)
    parser.add_argument("--effect_filter_mode", default="gt", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seed = set_seed(args.seed)
    device = get_device(args.gpu_id)
    output_dir = ensure_output_dir(args.output_dir, args.model_path)

    model_path = str(args.model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

    if args.eval_modes == "single":
        modes: List[Tuple[float, str]] = [(float(args.effect_threshold), str(args.effect_filter_mode))]
    else:
        modes = [(0.0, "gt"), (0.5, "gt"), (0.5, "lt")]

    loaded_model = None
    loaded_num_classes = None
    all_results = []

    for effect_threshold, effect_filter_mode in modes:
        val_loader, class_names, num_classes = build_val_loader(
            data_file=args.data_file,
            min_samples=int(args.min_samples),
            effect_size_file=args.effect_size_file,
            effect_threshold=float(effect_threshold),
            effect_filter_mode=str(effect_filter_mode),
            seed=int(seed),
            train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio),
            balance_strategy=str(args.balance_strategy),
            batch_size=int(args.batch_size),
        )

        if loaded_model is None:
            loaded_num_classes = int(num_classes)
            loaded_model = ImprovedGCN(
                num_features=1,
                hidden_dim=64,
                num_classes=int(num_classes),
                dropout=0.3,
            ).to(device)
            state_dict = torch.load(model_path, map_location=device)
            loaded_model.load_state_dict(state_dict, strict=True)
        else:
            if int(num_classes) != int(loaded_num_classes):
                raise ValueError(
                    f"不同模式下类别数不一致，无法复用同一个模型: {loaded_num_classes} vs {num_classes}"
                )

        metrics = evaluate_model(loaded_model, val_loader, device)
        report = classification_report(
            metrics["labels"],
            metrics["predictions"],
            target_names=class_names,
            zero_division=0,
        )
        cm = confusion_matrix(metrics["labels"], metrics["predictions"])

        result = {
            "effect_threshold": float(effect_threshold),
            "effect_filter_mode": str(effect_filter_mode),
            "val_accuracy": float(metrics["accuracy"]),
            "val_precision_weighted": float(metrics["precision"]),
            "val_recall_weighted": float(metrics["recall"]),
            "val_f1_weighted": float(metrics["f1"]),
        }
        all_results.append(result)

        tag = f"thr{effect_threshold}_{effect_filter_mode}"
        tag = tag.replace(".", "p")
        with open(os.path.join(output_dir, f"val_metrics_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        with open(os.path.join(output_dir, f"val_report_{tag}.txt"), "w", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False, indent=2))
            f.write("\n\n")
            f.write(report)

        np.savetxt(os.path.join(output_dir, f"val_confusion_matrix_{tag}.csv"), cm, delimiter=",", fmt="%d")

        print("\n" + "=" * 80)
        print(f"验证集评估 (effect_threshold={effect_threshold}, effect_filter_mode={effect_filter_mode})")
        print(f"模型: {model_path}")
        print(f"设备: {device}")
        print(f"balance_strategy: {args.balance_strategy}")
        print(f"Accuracy: {metrics['accuracy']:.6f}")
        print(f"Precision(weighted): {metrics['precision']:.6f}")
        print(f"Recall(weighted): {metrics['recall']:.6f}")
        print(f"F1(weighted): {metrics['f1']:.6f}")
        print("\n分类报告:")
        print(report)

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "-" * 80)
    print(f"结果已保存到: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()

