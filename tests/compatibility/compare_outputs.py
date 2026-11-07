"""Compare identities, missingness, annotations and all numerical output columns."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare(left, right, pattern):
    files = sorted(left.glob(pattern))
    assert files, f'No files matching {pattern}'
    assert [p.name for p in files] == sorted(p.name for p in right.glob(pattern)), 'Output file sets differ'
    report = {}
    for path in files:
        if path.name == 'synthetic.tsv':
            assert path.read_bytes() == (right / path.name).read_bytes(), 'Synthetic inputs differ'
        a = pd.read_csv(path, sep='\t')
        b = pd.read_csv(right / path.name, sep='\t')
        assert a.columns.equals(b.columns), path.name
        assert a.shape == b.shape, path.name
        assert a.isna().equals(b.isna()), f'{path.name}: missingness differs'
        columns = {}
        for col in a:
            if pd.api.types.is_numeric_dtype(a[col]) and not pd.api.types.is_bool_dtype(a[col]):
                # Same SciPy/NumPy and deterministic fits should agree closely.
                # 1e-8 relative allows rounding; 1e-10 absolute protects near zero.
                np.testing.assert_allclose(a[col], b[col], rtol=1e-8, atol=1e-10,
                                           equal_nan=True, err_msg=f'{path.name}: {col}')
                finite = np.isfinite(a[col]) & np.isfinite(b[col])
                columns[col] = float((a.loc[finite, col] - b.loc[finite, col]).abs().max()) if finite.any() else 0.
            else:
                assert a[col].fillna('').equals(b[col].fillna('')), f'{path.name}: {col}'
        report[path.name] = {'rows': len(a), 'columns': len(a.columns), 'max_absolute_difference': columns}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--pattern', default='*.tsv')
    args = parser.parse_args()
    print(json.dumps(compare(args.left, args.right, args.pattern), indent=2))
