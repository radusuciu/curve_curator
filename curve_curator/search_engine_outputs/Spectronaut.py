import numpy as np
import pandas as pd


def _to_parquet_name(col):
    """
    Spectronaut's parquet writer derives each parquet column name from the csv one by replacing
    every dot and every space with an underscore. 'PG.Quantity' becomes 'PG_Quantity', and a pivot
    report's '[1] CondA.PG.Quantity' becomes '[1]_CondA_PG_Quantity'.

    Parameters
    ----------
    col : str

    Returns
    -------
    col : str
    """
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
    def rename_quantity_columns(cols):
        """
        Renames a pivot report's quantity headers to 'Raw <run>'.

        A pivot report carries one quantity column per run, headed '[1] <run>.PG.Quantity' in the
        csv dialect and '[1]_<run>_PG_Quantity' in the parquet one. The leading '[n] ' index is
        optional. Anything else, the long report's plain 'PG.Quantity' included, passes through.
        """
        cols = cols.str.replace(r'^(?:\[\d+\]\s*)?(?P<run>.+)\.(?:PG|PEP)\.Quantity$', r'Raw \g<run>', regex=True)
        return cols.str.replace(r'^(?:\[\d+\]_)?(?P<run>.+)_(?:PG|PEP)_Quantity$', r'Raw \g<run>', regex=True)

    @staticmethod
    def is_long_report(df):
        return 'Condition' in df.columns

    @staticmethod
    def restructure_long_report(df, index, value_col):
        """
        Pivots a long report into one 'Raw <condition>' column per condition in the report.

        Spectronaut has already rolled the quantity up, so a group holds one value repeated over
        the report's rows and aggfunc='first' is a lookup rather than an aggregation.
        assert_constant_within is what guarantees that.

        Only the identity columns go into the pivot index, because pandas drops rows whose group
        key is NaN and annotations such as Genes are legitimately empty. They are merged back on
        the identity instead, taking the first row of each group.

        Parameters
        ----------
        df : pd.DataFrame
            long report with a 'Condition' column
        index : array-like
            column names that identify the level being parsed
        value_col : str
            name of the quantity column

        Returns
        -------
        df : pd.DataFrame
        """
        quantities = pd.pivot_table(data=df, values=value_col, index=index, columns='Condition', aggfunc='first', dropna=False)
        quantities = quantities.rename(columns=lambda c: f'Raw {c}')
        quantities.columns.name = None
        quantities.reset_index(inplace=True)
        annotations = df.drop(columns=[value_col, 'Condition']).drop_duplicates(subset=index)
        df = pd.merge(left=quantities, right=annotations, on=index, how='left')
        df.reset_index(drop=True, inplace=True)
        return df

    # Both quantity columns map to the same canonical name. The loaders read an explicit column
    # list, so only the one belonging to the level being parsed is ever present in the frame.
    _col_map = {
        'R.Condition': 'Condition',
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

    # Spectronaut writes real booleans in parquet and 'True' / 'False' strings in tsv. Anything
    # else, missing values included, is not a decoy.
    _indicator_map = {
        True: True,
        'True': True,
        False: False,
        'False': False,
        np.nan: False,
        '': False,
    }
