import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train  # noqa: E402


def make_df():
    rows = []
    for i in range(40):
        for j in range(3):
            rows.append({"Domain": f"site{i}.com", "Label": i % 2,
                         **{f: (i + j) % 2 for f in train.ALL_FEATURES}, "URL_Depth": j})
    return pd.DataFrame(rows)


def test_no_domain_overlap_between_train_and_test():
    tr, te = train.group_split(make_df())
    assert not set(tr["Domain"]) & set(te["Domain"])
    assert len(tr) and len(te)


def test_clean_drops_conflicting_domains_and_keeps_duplicates():
    df = pd.DataFrame({"Domain": ["a.com", "a.com", "b.com", "b.com", "c.com"],
                       "Label": [0, 1, 1, 1, 0],
                       **{f: [0, 0, 1, 1, 0] for f in train.ALL_FEATURES}})
    out = train.clean(df)
    assert "a.com" not in set(out["Domain"])       # conflicting labels removed
    assert (out["Domain"] == "b.com").sum() == 2   # exact duplicates are kept


def test_feature_lists_are_disjoint_and_complete():
    assert not set(train.URL_FEATURES) & set(train.LOOKUP_FEATURES)
    assert len(train.ALL_FEATURES) == 16


def test_models_fit_and_predict():
    df = make_df()
    for model in train.build_models().values():
        model.fit(df[train.ALL_FEATURES], df["Label"])
        assert len(model.predict(df[train.ALL_FEATURES])) == len(df)
