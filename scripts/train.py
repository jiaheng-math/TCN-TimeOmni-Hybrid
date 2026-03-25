from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.cmapss_dataset import build_dataloaders
from losses.gaussian_nll import gaussian_nll_loss, weighted_point_loss
from metrics.phm_score import compute_phm_score
from metrics.rmse import compute_rmse
from metrics.uncertainty_metrics import compute_mpiw, compute_picp
from models.tcn_rul_model import TCNPointModel, TCNUncertaintyModel
from utils.logger import append_results_summary, get_timestamp, save_history, save_json, setup_logger
from utils.rul import clip_rul_array
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TCN model on CMAPSS.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def ensure_output_dirs(config: dict) -> None:
    for key in ["results_dir", "figures_dir", "checkpoint_dir", "logs_dir"]:
        Path(config["output"][key]).mkdir(parents=True, exist_ok=True)


def build_model(config: dict, input_dim: int) -> torch.nn.Module:
    model_cfg = config["model"]
    common_kwargs = {
        "n_features": input_dim,
        "num_channels": model_cfg["num_channels"],
        "kernel_size": model_cfg["kernel_size"],
        "dropout": model_cfg["dropout"],
    }
    if model_cfg["type"] == "point":
        return TCNPointModel(**common_kwargs)
    if model_cfg["type"] == "uncertainty":
        return TCNUncertaintyModel(**common_kwargs)
    raise ValueError(f"Unsupported model type: {model_cfg['type']}")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def get_monitor_value(val_loss: float, val_rmse: float, monitor_name: str) -> float:
    if monitor_name == "val_loss":
        return val_loss
    if monitor_name == "val_rmse":
        return val_rmse
    raise ValueError(f"Unsupported monitor: {monitor_name}")


def maybe_clip_predictions(pred: np.ndarray, config: dict) -> np.ndarray:
    if not config["training"].get("clip_predictions", False):
        return pred
    return clip_rul_array(pred, min_value=0.0, max_value=float(config["data"]["rul_clip"]))


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    model_type: str,
    train: bool,
    config: dict,
    epoch: int | None = None,
    total_epochs: int | None = None,
    stage_name: str | None = None,
) -> dict:
    """执行一个 epoch 的训练或评估。

    统一处理 point 和 uncertainty 两种模型类型：
    - point: 前向传播 → MSE 损失
    - uncertainty: 前向传播 → (μ, logvar) → 高斯 NLL 损失

    返回包含 loss、RMSE、PHM score 的字典；
    uncertainty 模型额外返回 PICP、MPIW 等不确定性指标。
    """
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    preds = []
    targets = []
    mus = []
    logvars = []

    stage = stage_name or ("Train" if train else "Eval")
    epoch_desc = f"Epoch {epoch}/{total_epochs}" if epoch is not None and total_epochs is not None else "Inference"
    progress = tqdm(
        loader,
        desc=f"{epoch_desc} [{stage}]",
        leave=False,
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )

    for batch_idx, (x, y) in enumerate(progress, start=1):
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if model_type == "point":
                pred = model(x)
                loss = weighted_point_loss(
                    pred,
                    y,
                    loss_name=config["training"].get("point_loss", "mse"),
                    low_rul_threshold=config["training"].get("low_rul_threshold"),
                    low_rul_weight=float(config["training"].get("low_rul_weight", 1.0)),
                    smooth_l1_beta=float(config["training"].get("smooth_l1_beta", 1.0)),
                )
                batch_mu = pred
                batch_logvar = None
            else:
                batch_mu, batch_logvar = model(x)
                loss = gaussian_nll_loss(batch_mu, batch_logvar, y)

            if train:
                loss.backward()
                grad_clip_norm = config["training"].get("gradient_clip_norm")
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
                optimizer.step()

        losses.append(loss.item())
        preds.append(batch_mu.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())
        mus.append(batch_mu.detach().cpu().numpy())
        if batch_logvar is not None:
            logvars.append(batch_logvar.detach().cpu().numpy())

        progress.set_postfix(loss=f"{np.mean(losses):.4f}", batch=batch_idx)

    pred_arr = maybe_clip_predictions(np.concatenate(preds), config)
    true_arr = np.concatenate(targets)
    mu_arr = maybe_clip_predictions(np.concatenate(mus), config)
    metrics = {
        "loss": float(np.mean(losses)),
        "rmse": compute_rmse(pred_arr, true_arr),
        "phm_score": compute_phm_score(pred_arr, true_arr),
        "pred": pred_arr,
        "true": true_arr,
        "mu": mu_arr,
    }

    if logvars:
        logvar_arr = np.concatenate(logvars)
        sigma_arr = np.exp(0.5 * logvar_arr)  # σ = exp(0.5 * log(σ²))
        # 95% 置信区间：μ ± 1.96σ（正态分布双侧 95% 分位数）
        lower = mu_arr - 1.96 * sigma_arr
        upper = mu_arr + 1.96 * sigma_arr
        metrics.update(
            {
                "logvar": logvar_arr,
                "sigma_mean": float(np.mean(sigma_arr)),
                "sigma_std": float(np.std(sigma_arr)),
                "picp": compute_picp(lower, upper, true_arr),
                "mpiw": compute_mpiw(lower, upper),
                "lower": lower,
                "upper": upper,
            }
        )
    return metrics


def compute_engine_level_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    unit_ids: np.ndarray,
    cycles: np.ndarray,
) -> dict:
    """将窗口级预测聚合为发动机级指标（取每台发动机最后一个窗口的预测）。

    验证集使用滑动窗口（stride=1），每台发动机产生多个窗口预测；
    而测试集每台发动机只有一个窗口。为使 val / test 指标可比，
    val 也应取每台发动机最后一个周期的预测来计算 RMSE 和 PHM score。
    """
    engine_pred = []
    engine_true = []
    for uid in np.unique(unit_ids):
        mask = unit_ids == uid
        uid_cycles = cycles[mask]
        # 取该发动机最后一个周期（最大 cycle）对应的预测
        last_idx = np.argmax(uid_cycles)
        indices = np.where(mask)[0]
        engine_pred.append(pred[indices[last_idx]])
        engine_true.append(true[indices[last_idx]])
    engine_pred = np.asarray(engine_pred)
    engine_true = np.asarray(engine_true)
    return {
        "rmse": compute_rmse(engine_pred, engine_true),
        "phm_score": compute_phm_score(engine_pred, engine_true),
    }


def evaluate_on_test(model, loader, dataset, device, model_type: str, config: dict) -> dict:
    metrics = run_epoch(
        model=model,
        loader=loader,
        optimizer=None,
        device=device,
        model_type=model_type,
        train=False,
        config=config,
        stage_name="Test",
    )
    result = {
        "test_rmse": metrics["rmse"],
        "test_phm_score": metrics["phm_score"],
        "unit_ids": dataset.unit_ids.tolist(),
        "true_rul": dataset.y.cpu().numpy().tolist(),
        "pred_mu": metrics["mu"].tolist(),
    }
    if model_type == "uncertainty":
        result["test_picp"] = metrics["picp"]
        result["test_mpiw"] = metrics["mpiw"]
        result["lower"] = metrics["lower"].tolist()
        result["upper"] = metrics["upper"].tolist()
        result["sigma_mean"] = metrics["sigma_mean"]
        result["sigma_std"] = metrics["sigma_std"]
    return result


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_epoch: int,
    best_monitor_value: float,
    best_val_loss: float,
    best_val_rmse: float,
    epochs_without_improvement: int,
    input_dim: int,
    config: dict,
    history: list[dict],
) -> None:
    """保存完整的训练状态，支持断点续训。"""
    payload = {
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_monitor_value": best_monitor_value,
        "best_val_loss": best_val_loss,
        "best_val_rmse": best_val_rmse,
        "epochs_without_improvement": epochs_without_improvement,
        "input_dim": input_dim,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "config": config,
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    set_seed(config["training"]["seed"])

    subset = config["data"]["subset"]
    model_type = config["model"]["type"]
    timestamp = get_timestamp()
    logger = setup_logger(
        name=f"train_{subset}_{model_type}",
        log_dir=config["output"]["logs_dir"],
        filename=f"train_{subset}_{model_type}_{timestamp}.log",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading data for subset %s", subset)
    bundle = build_dataloaders(config)
    bundle.feature_processor.save(Path(config["output"]["checkpoint_dir"]) / f"scaler_{subset}_{model_type}.json")
    logger.info("Train units: %d | Val units: %d", len(bundle.train_units), len(bundle.val_units))
    logger.info("Removed sensors: %s", bundle.feature_processor.removed_sensor_columns)
    logger.info("Kept sensors: %s", bundle.feature_processor.kept_sensor_columns)
    logger.info("Input features: %s", bundle.feature_processor.feature_columns)
    logger.info(
        "Dataset sizes | train windows: %d | val windows: %d | test engines: %d",
        len(bundle.train_dataset),
        len(bundle.val_dataset),
        len(bundle.test_dataset),
    )

    model = build_model(config, bundle.input_dim).to(device)
    logger.info("Model parameters: %d", count_parameters(model))

    optimizer_name = config["training"]["optimizer"]
    weight_decay = config["training"].get("weight_decay", 0.0)
    if optimizer_name == "Adam":
        optimizer = Adam(model.parameters(), lr=config["training"]["lr"], weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=weight_decay)
    else:
        raise ValueError("Only Adam and AdamW optimizers are implemented.")

    if config["training"]["scheduler"] != "ReduceLROnPlateau":
        raise ValueError("Only ReduceLROnPlateau scheduler is implemented.")
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config["training"]["scheduler_patience"],
        factor=config["training"]["scheduler_factor"],
    )

    checkpoint_path = Path(config["output"]["checkpoint_dir"]) / f"best_model_{subset}_{model_type}.pth"
    # 断点续训用的 latest checkpoint（每个 epoch 都保存，包含完整训练状态）
    latest_checkpoint_path = Path(config["output"]["checkpoint_dir"]) / f"latest_model_{subset}_{model_type}.pth"
    history_path = Path(config["output"]["logs_dir"]) / f"history_{subset}_{model_type}.csv"
    train_summary_path = Path(config["output"]["logs_dir"]) / f"train_summary_{subset}_{model_type}.json"
    results_summary_path = Path(config["output"]["results_dir"]) / "results_summary.csv"

    monitor_name = config["training"].get("early_stopping_monitor", "val_loss")
    scheduler_monitor = config["training"].get("scheduler_monitor", monitor_name)

    best_monitor_value = math.inf
    best_val_loss = math.inf
    best_val_rmse = math.inf
    best_epoch = 0
    epochs_without_improvement = 0  # 早停计数器
    history = []
    start_epoch = 1

    # ========== 断点续训：从 latest checkpoint 恢复完整训练状态 ==========
    if args.resume and latest_checkpoint_path.exists():
        ckpt = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_epoch = ckpt.get("best_epoch", ckpt["epoch"])
        best_monitor_value = ckpt.get("best_monitor_value", math.inf)
        best_val_loss = ckpt["best_val_loss"]
        best_val_rmse = ckpt.get("best_val_rmse", math.inf)
        epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        history = ckpt.get("history", [])
        logger.info(
            "Resumed from epoch %d (best_val_loss=%.4f, patience=%d/%d)",
            ckpt["epoch"],
            best_val_loss,
            epochs_without_improvement,
            config["training"]["early_stopping_patience"],
        )
    elif args.resume:
        logger.info("No checkpoint found at %s, training from scratch.", latest_checkpoint_path)

    # ========== 训练循环 ==========
    total_epochs = config["training"]["epochs"]
    for epoch in range(start_epoch, total_epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=bundle.train_loader,
            optimizer=optimizer,
            device=device,
            model_type=model_type,
            train=True,
            config=config,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        val_metrics = run_epoch(
            model=model,
            loader=bundle.val_loader,
            optimizer=optimizer,
            device=device,
            model_type=model_type,
            train=False,
            config=config,
            epoch=epoch,
            total_epochs=total_epochs,
        )

        # 在发动机级别计算 val RMSE / PHM score，与 test 评估口径一致
        val_engine = compute_engine_level_metrics(
            val_metrics["pred"],
            val_metrics["true"],
            bundle.val_dataset.unit_ids,
            bundle.val_dataset.cycles,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_rmse": val_engine["rmse"],
            "val_phm_score": val_engine["phm_score"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        if model_type == "uncertainty":
            record["val_sigma_mean"] = val_metrics["sigma_mean"]
            record["val_sigma_std"] = val_metrics["sigma_std"]
        history.append(record)

        scheduler.step(get_monitor_value(val_metrics["loss"], val_engine["rmse"], scheduler_monitor))

        log_line = (
            f"Epoch {epoch}/{total_epochs} | Train Loss {train_metrics['loss']:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} | Val RMSE {val_engine['rmse']:.2f} | "
            f"Val Score {val_engine['phm_score']:.2f}"
        )
        if model_type == "uncertainty":
            log_line += (
                f" | Val Sigma Mean {val_metrics['sigma_mean']:.2f}"
                f" | Val Sigma Std {val_metrics['sigma_std']:.2f}"
            )
        logger.info(log_line)

        current_monitor = get_monitor_value(val_metrics["loss"], val_engine["rmse"], monitor_name)
        if current_monitor < best_monitor_value:
            best_monitor_value = current_monitor
            best_val_loss = val_metrics["loss"]
            best_val_rmse = val_engine["rmse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            # 保存最佳模型（用于最终评估）
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler,
                epoch, best_epoch, best_monitor_value, best_val_loss, best_val_rmse,
                epochs_without_improvement, bundle.input_dim, config, history,
            )
        else:
            epochs_without_improvement += 1

        # 每个 epoch 都保存 latest checkpoint（用于断点续训）
        save_checkpoint(
            latest_checkpoint_path, model, optimizer, scheduler,
            epoch, best_epoch, best_monitor_value, best_val_loss, best_val_rmse,
            epochs_without_improvement, bundle.input_dim, config, history,
        )

        if epochs_without_improvement >= config["training"]["early_stopping_patience"]:
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    # ========== 训练结束，保存结果并在测试集上评估 ==========
    save_history(history, history_path)
    save_json(
        {
            "subset": subset,
            "model_type": model_type,
            "best_epoch": best_epoch,
            "best_monitor": monitor_name,
            "best_monitor_value": best_monitor_value,
            "best_val_loss": best_val_loss,
            "best_val_rmse": best_val_rmse,
            "history_path": str(history_path),
            "checkpoint_path": str(checkpoint_path),
        },
        train_summary_path,
    )

    # 加载最佳 checkpoint（而非最后一个 epoch 的模型）进行测试
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_result = evaluate_on_test(model, bundle.test_loader, bundle.test_dataset, device, model_type, config)

    result_record = {
        "timestamp": timestamp,
        "subset": subset,
        "model_type": model_type,
        "best_epoch": best_epoch,
        "best_monitor": monitor_name,
        "best_monitor_value": best_monitor_value,
        "best_val_loss": best_val_loss,
        "best_val_rmse": best_val_rmse,
        "test_rmse": test_result["test_rmse"],
        "test_phm_score": test_result["test_phm_score"],
    }
    if model_type == "uncertainty":
        result_record["test_picp"] = test_result["test_picp"]
        result_record["test_mpiw"] = test_result["test_mpiw"]
    append_results_summary(result_record, results_summary_path)

    test_log_path = Path(config["output"]["logs_dir"]) / f"test_metrics_{subset}_{model_type}_{timestamp}.json"
    save_json(test_result, test_log_path)

    # ========== 训练总结 ==========
    logger.info("=" * 60)
    logger.info("Training complete")
    logger.info("  Best epoch       : %d / %d", best_epoch, total_epochs)
    logger.info("  Best val loss    : %.4f", best_val_loss)
    logger.info("  Best val RMSE    : %.4f", best_val_rmse)
    logger.info("  Checkpoint       : %s", checkpoint_path)
    logger.info("-" * 60)
    logger.info("  Test RMSE        : %.4f", test_result["test_rmse"])
    logger.info("  Test PHM Score   : %.4f", test_result["test_phm_score"])
    if model_type == "uncertainty":
        logger.info("  Test PICP        : %.4f", test_result["test_picp"])
        logger.info("  Test MPIW        : %.4f", test_result["test_mpiw"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
