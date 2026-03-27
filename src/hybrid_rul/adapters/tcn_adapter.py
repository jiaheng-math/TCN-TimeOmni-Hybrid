from __future__ import annotations

import copy
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from hybrid_rul.paths import load_yaml


class TCNProjectAdapter:
    def __init__(
        self,
        repo_root: str | Path,
        config_path: str | Path,
        module_root: str | Path | None = None,
        project_root: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        prediction_artifact_path: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.config_path = Path(config_path).resolve()
        self.module_root = Path(module_root).resolve() if module_root else self.repo_root
        self.project_root = Path(project_root).resolve() if project_root else self.repo_root
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None
        self.prediction_artifact_path = Path(prediction_artifact_path).resolve() if prediction_artifact_path else None
        self.device = self._resolve_device(device)

        self.config: dict | None = None
        self.bundle = None
        self.model = None
        self.model_type: str | None = None
        self.experiment_name: str | None = None
        self.sigma_scale: float | None = None
        self.prediction_artifact: Path | None = None
        self.predictions: list[dict] | None = None

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _ensure_repo_on_path(self) -> None:
        repo_str = str(self.module_root)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    def _import_original_modules(self):
        self._ensure_repo_on_path()
        dataset_module = importlib.import_module("datasets.cmapss_dataset")
        model_module = importlib.import_module("models")
        experiment_module = importlib.import_module("utils.experiment")
        return dataset_module, model_module, experiment_module

    def _normalize_config_paths(self, config: dict) -> dict:
        normalized = copy.deepcopy(config)
        data_cfg = normalized.get("data", {})
        if "data_dir" in data_cfg:
            data_cfg["data_dir"] = str((self.project_root / data_cfg["data_dir"]).resolve())

        output_cfg = normalized.get("output", {})
        for key in ("results_dir", "figures_dir", "checkpoint_dir", "logs_dir"):
            if key in output_cfg:
                output_cfg[key] = str((self.project_root / output_cfg[key]).resolve())
        return normalized

    def load(self) -> None:
        dataset_module, model_module, experiment_module = self._import_original_modules()

        raw_config = load_yaml(self.config_path)
        self.config = self._normalize_config_paths(raw_config)
        self.model_type = self.config["model"]["type"]

        self.bundle = dataset_module.build_dataloaders(self.config)

        experiment_name = experiment_module.get_experiment_name(self.config, self.config_path)
        self.experiment_name = experiment_name
        self.prediction_artifact = self.prediction_artifact_path
        if self.prediction_artifact is not None:
            return

        self.model = model_module.build_model(self.config, self.bundle.input_dim).to(self.device)

        checkpoint_path = self.checkpoint_path
        if checkpoint_path is None:
            checkpoint_dir = Path(self.config["output"]["checkpoint_dir"])
            checkpoint_path = checkpoint_dir / f"best_model_{experiment_name}.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"TCN checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.sigma_scale = None
        if self.model_type == "uncertainty":
            sigma_scale_path = Path(self.config["output"]["checkpoint_dir"]) / f"sigma_scale_{experiment_name}.json"
            if sigma_scale_path.exists():
                with sigma_scale_path.open("r", encoding="utf-8") as fp:
                    self.sigma_scale = float(json.load(fp)["sigma_scale"])

    def _maybe_clip_predictions(self, values: np.ndarray) -> np.ndarray:
        assert self.config is not None
        if not self.config["training"].get("clip_predictions", False):
            return values
        max_value = float(self.config["data"]["rul_clip"])
        return np.clip(values, 0.0, max_value)

    def _point_warning(self, mu: float) -> dict:
        assert self.config is not None
        thresholds = self.config["warning"]["thresholds"]
        if mu > thresholds["normal"]:
            level = "正常"
        elif mu > thresholds["watch"]:
            level = "关注"
        elif mu > thresholds["alert"]:
            level = "预警"
        else:
            level = "危险"
        return {
            "level": level,
            "sigma": None,
            "lower": None,
            "escalated": False,
        }

    def _warning_from_sigma(self, mu: float, sigma: float) -> dict:
        assert self.config is not None
        warning_cfg = self.config["warning"]
        thresholds = warning_cfg["thresholds"]
        sigma_threshold = float(warning_cfg["sigma_threshold"])
        sigma_escalation = bool(warning_cfg.get("sigma_escalation", True))
        levels = ["正常", "关注", "预警", "危险"]

        lower = float(mu) - 1.96 * float(sigma)
        if lower > thresholds["normal"]:
            level_idx = 0
        elif lower > thresholds["watch"]:
            level_idx = 1
        elif lower > thresholds["alert"]:
            level_idx = 2
        else:
            level_idx = 3

        escalated = False
        if sigma_escalation and sigma > sigma_threshold and level_idx < len(levels) - 1:
            level_idx += 1
            escalated = True

        return {
            "level": levels[level_idx],
            "sigma": float(sigma),
            "lower": float(lower),
            "escalated": escalated,
        }

    def _load_predictions_from_artifact(self) -> list[dict]:
        assert self.bundle is not None
        assert self.config is not None
        assert self.prediction_artifact is not None

        with self.prediction_artifact.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)

        unit_ids = [int(value) for value in payload["unit_ids"]]
        pred_mu = np.asarray(payload["pred_mu"], dtype=np.float64)
        true_rul = np.asarray(payload["true_rul"], dtype=np.float64)
        lower_values = payload.get("lower")
        upper_values = payload.get("upper")
        lower = np.asarray(lower_values, dtype=np.float64) if lower_values is not None else None
        upper = np.asarray(upper_values, dtype=np.float64) if upper_values is not None else None

        cycle_map = {
            int(unit_id): int(cycle)
            for unit_id, cycle in zip(self.bundle.test_dataset.unit_ids.tolist(), self.bundle.test_dataset.cycles.tolist())
        }

        predictions: list[dict] = []
        for idx, unit_id in enumerate(unit_ids):
            mu_value = float(pred_mu[idx])
            base_payload = {
                "unit_id": unit_id,
                "observed_cycle": cycle_map[unit_id],
                "predicted_rul": mu_value,
                "true_rul": float(true_rul[idx]),
                "model_type": self.model_type,
            }

            if lower is not None and upper is not None:
                lower_value = float(lower[idx])
                upper_value = float(upper[idx])
                sigma_value = max((upper_value - lower_value) / 3.92, 0.0)
                warning = self._warning_from_sigma(mu_value, sigma_value)
                warning["lower"] = lower_value
                base_payload.update(
                    {
                        "sigma": sigma_value,
                        "lower_95": lower_value,
                        "upper_95": upper_value,
                        "warning": warning,
                    }
                )
            else:
                base_payload["warning"] = self._point_warning(mu_value)

            predictions.append(base_payload)

        return predictions

    def predict_test_set(self) -> list[dict]:
        needs_model = self.prediction_artifact is None
        if self.bundle is None or self.config is None or (needs_model and self.model is None):
            self.load()

        assert self.bundle is not None
        assert self.config is not None

        if self.prediction_artifact is not None:
            self.predictions = self._load_predictions_from_artifact()
            return self.predictions

        assert self.model is not None

        mu_batches = []
        logvar_batches = []
        true_batches = []

        with torch.no_grad():
            for x, y in self.bundle.test_loader:
                x = x.to(self.device)
                if self.model_type == "point":
                    mu = self.model(x)
                    logvar = None
                else:
                    mu, logvar = self.model(x)
                mu_batches.append(mu.detach().cpu().numpy())
                true_batches.append(y.numpy())
                if logvar is not None:
                    logvar_batches.append(logvar.detach().cpu().numpy())

        pred_mu = self._maybe_clip_predictions(np.concatenate(mu_batches))
        true_rul = np.concatenate(true_batches)
        unit_ids = self.bundle.test_dataset.unit_ids.astype(int)
        cycles = self.bundle.test_dataset.cycles.astype(int)

        predictions = []
        if logvar_batches:
            pred_logvar = np.concatenate(logvar_batches)
        else:
            pred_logvar = None

        for idx, unit_id in enumerate(unit_ids.tolist()):
            mu_value = float(pred_mu[idx])
            base_payload = {
                "unit_id": unit_id,
                "observed_cycle": int(cycles[idx]),
                "predicted_rul": mu_value,
                "true_rul": float(true_rul[idx]),
                "model_type": self.model_type,
            }

            if pred_logvar is not None:
                logvar_value = float(pred_logvar[idx])
                sigma_value = math.exp(0.5 * logvar_value)
                if self.sigma_scale is not None:
                    sigma_value *= self.sigma_scale
                lower = mu_value - 1.96 * sigma_value
                upper = mu_value + 1.96 * sigma_value
                warning = self._warning_from_sigma(mu_value, sigma_value)
                base_payload.update(
                    {
                        "logvar": logvar_value,
                        "sigma": sigma_value,
                        "lower_95": float(lower),
                        "upper_95": float(upper),
                        "warning": warning,
                    }
                )
            else:
                base_payload["warning"] = self._point_warning(mu_value)

            predictions.append(base_payload)

        self.predictions = predictions
        return predictions

    def get_unit_frame(self, unit_id: int):
        if self.bundle is None:
            self.load()
        assert self.bundle is not None
        return self.bundle.test_df[self.bundle.test_df["unit_id"] == unit_id].copy()

    def get_feature_columns(self) -> list[str]:
        if self.bundle is None:
            self.load()
        assert self.bundle is not None
        return list(self.bundle.feature_processor.feature_columns)
