"""Project settings and the training-window contract."""

import hashlib
from datetime import date
from pathlib import Path
from typing import Self

from pydantic import BaseModel, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from credit_mlops.months import add_months, month_label, months_between


class Settings(BaseSettings):
    """Priority: init kwargs > env vars > .env > config/settings.toml."""

    model_config = SettingsConfigDict(
        env_prefix="CREDIT_MLOPS_",
        env_file=".env",
        toml_file="config/settings.toml",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    seed: int = 42
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_registry_uri: str | None = None
    mlflow_artifact_root: str | None = None
    training_experiment: str = "credit-default-training"
    monitoring_experiment: str = "credit-default-monitoring"
    registered_model_name: str = "credit_default_pd"
    decision_threshold: float = 0.15
    explanation_top_k: int = 3
    hpo_n_estimators: list[int] = [200, 400, 800]
    hpo_max_depth: list[int] = [3, 5, 7]
    learning_rate: float = 0.05
    reference_sample_size: int = 5000
    psi_bins: int = 10
    psi_watch: float = 0.10
    psi_act: float = 0.25
    calibration_tolerance: float = 0.15
    gate_min_auc_gain: float = 0.0
    gate_max_region_auc_drop: float = 0.02
    monitor_max_auc_drop: float = 0.03
    monitor_pool_cohorts: int = 3
    region_min_loans: int = 500
    label_window_months: int = 12
    retrain_after_consecutive_alerts: int = 6
    retrain_half_life_months: float = 3.0
    production_start: date = date(2023, 1, 1)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    @model_validator(mode="after")
    def _check_uc_model_name(self) -> Self:
        if (
            self.mlflow_registry_uri == "databricks-uc"
            and self.registered_model_name.count(".") != 2
        ):
            raise ValueError(
                "with databricks-uc, registered_model_name must be 'catalog.schema.model'"
            )
        return self

    @property
    def applications_dir(self) -> Path:
        return self.data_dir / "raw" / "applications"

    @property
    def outcomes_path(self) -> Path:
        return self.data_dir / "raw" / "outcomes.parquet"

    @property
    def predictions_dir(self) -> Path:
        return self.data_dir / "predictions"

    def settings_hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]


class TrainingWindow(BaseModel):
    train_start: date
    train_end: date
    oot_start: date
    oot_end: date
    as_of: date
    validation_months: int = 6
    label_window_months: int = 12
    half_life_months: float | None = None

    @model_validator(mode="after")
    def _check_window(self) -> Self:
        for name in ("train_start", "train_end", "oot_start", "oot_end", "as_of"):
            if getattr(self, name).day != 1:
                raise ValueError(f"{name} must be the first day of a month")
        if self.train_start > self.train_end:
            raise ValueError("train_start must not be after train_end")
        if self.validation_months >= months_between(self.train_start, self.train_end) + 1:
            raise ValueError("validation_months must be shorter than the training period")
        if not (self.train_end < self.oot_start <= self.oot_end):
            raise ValueError("require train_end < oot_start <= oot_end (no overlap)")
        if add_months(self.oot_end, self.label_window_months) > self.as_of:
            raise ValueError("OOT labels are not mature at as_of")
        return self

    @property
    def validation_start(self) -> date:
        return add_months(self.train_end, -(self.validation_months - 1))

    @classmethod
    def rolled(
        cls,
        as_of: date,
        *,
        train_months: int = 12,
        oot_months: int = 3,
        validation_months: int = 3,
        half_life_months: float = 3.0,
        label_window_months: int = 12,
    ) -> Self:
        latest_matured = add_months(as_of, -label_window_months)
        oot_start = add_months(latest_matured, -(oot_months - 1))
        train_end = add_months(oot_start, -1)
        return cls(
            train_start=add_months(train_end, -(train_months - 1)),
            train_end=train_end,
            oot_start=oot_start,
            oot_end=latest_matured,
            as_of=as_of,
            validation_months=validation_months,
            label_window_months=label_window_months,
            half_life_months=half_life_months,
        )

    def as_params(self) -> dict[str, str]:
        """Flat string params (months as YYYY-MM) for MLflow logging."""
        return {
            "train_start": month_label(self.train_start),
            "train_end": month_label(self.train_end),
            "oot_start": month_label(self.oot_start),
            "oot_end": month_label(self.oot_end),
            "as_of": month_label(self.as_of),
            "validation_months": str(self.validation_months),
            "label_window_months": str(self.label_window_months),
            "half_life_months": str(self.half_life_months),
            "validation_start": month_label(self.validation_start),
        }


DEFAULT_WINDOW = TrainingWindow(
    train_start=date(2019, 1, 1),
    train_end=date(2021, 6, 1),
    oot_start=date(2021, 7, 1),
    oot_end=date(2021, 12, 1),
    as_of=date(2023, 1, 1),
)
