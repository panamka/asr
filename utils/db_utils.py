from numpy import log10
import torch


def to_db(ratio):
    assert ratio >= 0
    ratio_db = 10. * log10(ratio + 1e-8)
    return ratio_db

def from_db(ratio_db, base=10):
    ratio = 10 ** (ratio_db / base) - 1e-8
    return ratio

def get_coef(db_cur, db_target):
    diff = db_target - db_cur
    mult = from_db(diff / 2, base=10)
    return mult

def normal(mean=0, std=1, *, generator=None):
    x = torch.empty(1).normal_(mean, std, generator=generator).item()
    return x

