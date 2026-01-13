from __future__ import annotations

from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass
class HFDataSpec:
    """
    Simple specification for a Hugging Face dataset.
    """
    dataset_id: str
    subset: str | None = None
    split: str = "train"
    text_col: str = "sentence"
    label_col: str = "label"
    language: str = "xx"
    dataset_name: str = "hf"


def load_sample_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = df["label"].astype(int)
    return df


def load_hf_dataset(spec: HFDataSpec, max_samples: int | None = None, seed: int = 0) -> pd.DataFrame:
    """
    Loads a dataset from Hugging Face `datasets`.
    Note: requires internet the first time to download and cache the dataset.
    """
    from datasets import load_dataset

    ds = load_dataset(spec.dataset_id, spec.subset, split=spec.split) if spec.subset else load_dataset(spec.dataset_id, split=spec.split)
    df = ds.to_pandas()

    df = df[[spec.text_col, spec.label_col]].rename(columns={spec.text_col: "sentence", spec.label_col: "label"})
    df["label"] = df["label"].astype(int)
    df["language"] = spec.language
    df["dataset"] = spec.dataset_name

    if max_samples is not None and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    return df


def default_multilingual_specs() -> List[HFDataSpec]:
    """
    A pragmatic multilingual acceptability suite using readily available datasets:
    - English CoLA in GLUE
    - Russian RuCoLA
    - Italian ItaCoLA

    You can add more languages (e.g., JCoLA) by extending this list.
    """
    return [
        HFDataSpec(dataset_id="nyu-mll/glue", subset="cola", split="validation", text_col="sentence", label_col="label", language="en", dataset_name="CoLA"),
        HFDataSpec(dataset_id="RussianNLP/rucola", subset=None, split="test", text_col="sentence", label_col="label", language="ru", dataset_name="RuCoLA"),
        HFDataSpec(dataset_id="gsarti/itacola", subset=None, split="test", text_col="sentence", label_col="label", language="it", dataset_name="ItaCoLA"),
    ]
