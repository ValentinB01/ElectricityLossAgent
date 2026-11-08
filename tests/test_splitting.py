import numpy as np

from src.data_loader import ConsumerData, load_dataset
from src.splitting import make_splits
import pandas as pd


def test_stratified_disjoint_reproducible_and_order_invariant(synthetic_csv):
    path, config = synthetic_csv
    data = load_dataset(path, config)
    a, b = make_splits(data, config), make_splits(data, config)
    assert {name: len(ix) for name, ix in a.items()} == {"train": 28, "val": 6, "test": 6}
    assert len(set(np.concatenate(list(a.values())))) == len(data.ids)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
        assert data.labels[a[name]].mean() == .5
    reverse = np.arange(len(data.ids))[::-1]
    reordered = ConsumerData(data.ids[reverse], data.labels[reverse], data.values[reverse], data.dates, {})
    c = make_splits(reordered, config)
    for name in a:
        np.testing.assert_array_equal(data.ids[a[name]], reordered.ids[c[name]])


def test_imbalanced_stratification(config):
    labels = np.r_[np.ones(85, dtype=np.int8), np.zeros(915, dtype=np.int8)]
    data = ConsumerData(np.array([f"id{i:04d}" for i in range(1000)]), labels,
                        np.zeros((1000, 2)), pd.date_range("2014-01-01", periods=2), {})
    splits = make_splits(data, config)
    for ix in splits.values():
        assert abs(labels[ix].mean() - .085) < .01
