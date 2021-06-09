from __future__ import absolute_import
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score

from ..utils import to_numpy


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask




def mean_ap(distmat, query_ids=None, gallery_ids=None):
    distmat = to_numpy(distmat)
    m, n = distmat.shape

    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)

    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis]).astype(np.int32)

    # Compute AP for each query
    aps = []
    for i in range(m):
        match = matches[i]
        match_cs = match.cumsum()
        prec = match_cs / np.arange(1,n+1)
        res = prec * match
        ap = res.sum()/ match.sum()
        aps.append(ap)
             
    return np.mean(aps)
