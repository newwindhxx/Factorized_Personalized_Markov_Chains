# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 13:06:41 2021

@author: lpott
"""

import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm


class ColumnView(object):
    def __init__(self, values):
        self.values = values

    def nunique(self):
        return int(np.unique(self.values).size)

    def unique(self):
        return np.unique(self.values)

    def min(self):
        return self.values.min()

    def to_numpy(self):
        return self.values

    def tolist(self):
        return self.values.tolist()


class RatingsData(object):
    columns = ("user_id", "item_id", "rating", "timestamp")

    def __init__(self, user_id, item_id, rating, timestamp):
        self.user_id = np.asarray(user_id, dtype=np.int64)
        self.item_id = np.asarray(item_id, dtype=np.int64)
        self.rating = np.asarray(rating, dtype=np.int64)
        self.timestamp = np.asarray(timestamp, dtype=np.int64)

    def __len__(self):
        return len(self.user_id)

    def __getitem__(self, key):
        if key not in self.columns:
            raise KeyError(key)
        return ColumnView(getattr(self, key))

    @property
    def shape(self):
        return (len(self), len(self.columns))

    def copy(self):
        return RatingsData(
            self.user_id.copy(),
            self.item_id.copy(),
            self.rating.copy(),
            self.timestamp.copy(),
        )

    def sort_values(self, column, inplace=False):
        order = np.argsort(getattr(self, column), kind="mergesort")
        target = self if inplace else self.copy()
        for name in self.columns:
            setattr(target, name, getattr(target, name)[order])
        return None if inplace else target

    def reset_index(self, drop=True):
        return self

    def nunique(self):
        return {name: int(np.unique(getattr(self, name)).size) for name in self.columns}

    def head(self, n=5):
        return [
            {name: int(getattr(self, name)[idx]) for name in self.columns}
            for idx in range(min(n, len(self)))
        ]


def create_df(filename=None):
    print("="*10,"Creating DataFrame","="*10)

    records = [[], [], [], []]
    with open(os.path.join(os.getcwd(), filename), "r", encoding="latin-1") as f:
        for line in f:
            user_id, item_id, rating, timestamp = line.rstrip("\n").split("::")
            records[0].append(int(user_id))
            records[1].append(int(item_id))
            records[2].append(int(rating))
            records[3].append(int(timestamp))

    df = RatingsData(*records)
    df.sort_values('timestamp', inplace=True)

    print(df.nunique())
    print(df.shape)

    return df.reset_index(drop=True)


class reset_df(object):
    def __init__(self):
        print("="*10,"Initialize Reset DataFrame Object","="*10)
        self.item_classes_ = None
        self.user_classes_ = None

    def fit_transform(self, df):
        print("="*10,"Resetting user ids and item ids in DataFrame","="*10)
        df = df.copy()

        self.item_classes_, item_codes = np.unique(df.item_id, return_inverse=True)
        self.user_classes_, user_codes = np.unique(df.user_id, return_inverse=True)
        df.item_id = item_codes.astype(np.int64)
        df.user_id = user_codes.astype(np.int64)

        assert df.user_id.min() == 0
        assert df.item_id.min() == 0

        return df

    def inverse_transform(self, df):
        df = df.copy()
        df.item_id = self.item_classes_[df.item_id]
        df.user_id = self.user_classes_[df.user_id]
        return df


def create_user_history(df=None):
    if df is None:
        return None

    print("="*10,"Creating User Histories","="*10)

    user_history = defaultdict(list)
    for uid, iid in tqdm(zip(df.user_id, df.item_id), total=len(df)):
        user_history[int(uid)].append(int(iid))

    return dict(user_history)


def train_val_test_split(user_history=None):
    if user_history is None:
        return None

    print("="*10,"Splitting User Histories into Train, Validation, and Test Splits","="*10)
    train_history = []
    val_history = []
    test_history = []

    for key, history in tqdm(user_history.items(), position=0, leave=True):
        if len(history) < 5:
            continue

        pairs = [(key, history[t], history[t + 1]) for t in range(len(history) - 1)]
        train_history.extend(pairs[:-2])
        val_history.append(pairs[-2])
        test_history.append(pairs[-1])

    return train_history, val_history, test_history


def create_user_noclick(user_history, df, n_items):
    print("="*10,"Creating User 'no-click' history","="*10)
    user_noclick = {}
    all_items = np.arange(n_items)
    item_counts = np.bincount(df.item_id, minlength=n_items).astype(np.float64)

    for uid, history in tqdm(user_history.items()):
        no_clicks = np.setdiff1d(all_items, np.asarray(history, dtype=np.int64), assume_unique=False)
        weights = item_counts[no_clicks]
        if weights.sum() == 0:
            probabilities = np.full(len(no_clicks), 1.0 / len(no_clicks))
        else:
            probabilities = weights / weights.sum()

        user_noclick[uid] = (no_clicks.tolist(), probabilities)

    return user_noclick
