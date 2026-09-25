import re

import numpy as np
import pandas as pd


def _to_parquet_name(col):
    """Convert a Spectronaut CSV column name to its Parquet form."""
    return col.replace('.', '_').replace(' ', '_')


class SpectronautMap:
    def __init__(self, version):
        version = version.split('.')
        self.version_major = version[0]
        self.version_minor = version[1]

    @staticmethod
    def rename_general_columns(cols):
        return cols.map(lambda c: SpectronautMap._col_map.get(c, SpectronautMap._parquet_col_map.get(c, c)))

    @staticmethod
    def map_indicator_values(df):
        for col in SpectronautMap._indicator_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda v: SpectronautMap._indicator_map.get(v, False))
        return df

    @staticmethod
    def rename_quantity_columns(cols, experiments=None, parquet=None):
        """Rename pivot quantity headers to ``Raw <run>`` in either dialect."""
        csv_pattern = re.compile(r'^(?:\[\d+\]\s*)?(?P<run>.+)\.(?:PG|PEP)\.Quantity$')
        parquet_pattern = re.compile(r'^(?:\[\d+\]_)?(?P<run>.+)_(?:PG|PEP)_Quantity$')
        renamed = []
        parquet_headers = []

        for col in cols:
            value = str(col)
            match = csv_pattern.match(value) if parquet is not True else None
            if match:
                renamed.append(f"Raw {match.group('run')}")
                continue

            match = parquet_pattern.match(value) if parquet is not False else None
            if match:
                token = match.group('run')
                renamed.append(f'Raw {token}')
                parquet_headers.append((len(renamed) - 1, value, token))
                continue

            renamed.append(col)

        if not parquet_headers or experiments is None:
            return pd.Index(renamed)

        configured = {}
        for experiment in experiments:
            experiment = str(experiment)
            configured.setdefault(_to_parquet_name(experiment), []).append(experiment)

        configured_collisions = {token: names for token, names in configured.items() if len(names) > 1}
        if configured_collisions:
            details = '; '.join(f"{names} -> {token!r}" for token, names in configured_collisions.items())
            raise ValueError(
                'The configured Spectronaut experiment names are ambiguous in Parquet headers: '
                f'{details}. Dots and spaces are both replaced with underscores.')

        reported = {}
        for _, header, token in parquet_headers:
            reported.setdefault(token, []).append(header)
        report_collisions = {token: headers for token, headers in reported.items() if len(headers) > 1}
        if report_collisions:
            details = '; '.join(f"{headers} -> {token!r}" for token, headers in report_collisions.items())
            raise ValueError(
                'The Spectronaut Parquet quantity headers are ambiguous after normalization: '
                f'{details}. The report must contain one distinct quantity header per run.')

        configured = {token: names[0] for token, names in configured.items()}
        for position, _, token in parquet_headers:
            if token in configured:
                renamed[position] = f'Raw {configured[token]}'
        return pd.Index(renamed)

    @staticmethod
    def is_long_report(df):
        return 'Run' in df.columns

    @staticmethod
    def restructure_long_report(df, index, value_col):
        """Pivot a long report into one ``Raw <run>`` column per run."""
        quantities = pd.pivot_table(data=df, values=value_col, index=index, columns='Run', aggfunc='first', dropna=False)
        quantities = quantities.rename(columns=lambda c: f'Raw {c}')
        quantities.columns.name = None
        quantities.reset_index(inplace=True)
        annotations = df.drop(columns=[value_col, 'Run']).drop_duplicates(subset=index)
        df = pd.merge(left=quantities, right=annotations, on=index, how='left')
        df.reset_index(drop=True, inplace=True)
        return df

    # The loader selects one level before both quantities map to the same name.
    _col_map = {
        'R.Label': 'Run',
        'PG.ProteinGroups': 'Proteins',
        'PG.Genes': 'Genes',
        'PEP.GroupingKey': 'Modified sequence',
        'PEP.GroupingKeyType': 'Grouping type',
        'EG.IsDecoy': 'Decoy',
        'PG.Quantity': 'Quantity',
        'PEP.Quantity': 'Quantity',
    }

    _parquet_col_map = {_to_parquet_name(col): name for col, name in _col_map.items()}

    _indicator_cols = ['Decoy']

    # Parquet uses booleans; TSV uses strings.
    _indicator_map = {
        True: True,
        'True': True,
        False: False,
        'False': False,
        np.nan: False,
        '': False,
    }
