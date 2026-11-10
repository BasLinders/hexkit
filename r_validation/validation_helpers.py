"""Tiny comparison helper shared by the R validation notebooks.

Each notebook collects (hexkit value, R value) pairs in a `Checks` object and
prints a pass/fail table at the end. `report()` raises if anything failed, so
running a notebook top-to-bottom (e.g. `jupyter nbconvert --execute`) doubles
as a test.
"""

import math

import pandas as pd


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, name, hexkit, r, atol=1e-8, rtol=0.0):
        """Pass if |hexkit - r| <= atol + rtol * |r|. Infinities must match exactly."""
        hexkit, r = float(hexkit), float(r)
        if math.isinf(hexkit) or math.isinf(r):
            ok = hexkit == r
            diff = 0.0 if ok else math.inf
        else:
            diff = abs(hexkit - r)
            ok = diff <= atol + rtol * abs(r)
        self.rows.append(
            {"check": name, "hexkit": hexkit, "R": r, "abs_diff": diff,
             "tolerance": f"atol={atol:g}, rtol={rtol:g}", "pass": ok}
        )
        return ok

    def add_equal(self, name, hexkit, r):
        """Pass if the two values are exactly equal (labels, decisions, counts)."""
        self.rows.append(
            {"check": name, "hexkit": hexkit, "R": r, "abs_diff": None,
             "tolerance": "exact", "pass": hexkit == r}
        )
        return hexkit == r

    def table(self):
        return pd.DataFrame(self.rows)

    def report(self):
        df = self.table()
        n_fail = int((~df["pass"]).sum())
        print(f"{len(df)} checks, {len(df) - n_fail} passed, {n_fail} failed")
        with pd.option_context("display.max_rows", None, "display.width", 200,
                               "display.float_format", "{:.6g}".format):
            failed = df[~df["pass"]]
            if n_fail:
                print(failed.to_string())
            numeric = df[df["abs_diff"].notna()].copy()
            if not numeric.empty:
                numeric["abs_diff"] = numeric["abs_diff"].astype(float)
                print("\nLargest differences:")
                print(numeric.nlargest(5, "abs_diff")[["check", "hexkit", "R", "abs_diff", "tolerance"]]
                      .to_string(index=False))
        if n_fail:
            raise AssertionError(f"{n_fail} check(s) did not match R")
        return df
