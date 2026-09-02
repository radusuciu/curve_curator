# data_parser.py
# Functions for harmonizing all sorts of input when reading in user data.
#
# Florian P. Bayer - 2025
#

import numpy as np
import pandas as pd

from .search_engine_outputs.MaxQuant import MaxQuantMap
from .search_engine_outputs.DIANN import DiannMap
from .search_engine_outputs.ProteomeDiscoverer import PDMap
from .search_engine_outputs.MSFragger import FraggerMap
from .search_engine_outputs.Spectronaut import SpectronautMap, _to_parquet_name
from . import user_interface as ui


def clean_modified_sequence(mod_seq):
    """
    Cleans up the modified sequence. Columns-wise operation is 10x faster than apply.

    Parameters
    ----------
    mod_seq : pd.Series(<seqs>)

    Returns
    -------
    mod_seq : pd.Series(<seqs>)
    """
    mappings = {
        r'_': '',
        r'\(\)': '',
        r'\(Acetyl \(Protein N-term\)\)':  '(ac)',
        r'M\(Oxidation \(M\)\)': 'M',
        r'M\(ox\)': 'M',
        r'M\(UniMod\:35\)': 'M',
        r'pS': 'S(ph)',
        r'pT': 'T(ph)',
        r'pY': 'Y(ph)',
        r'\(Phospho \(STY\)\)': '(ph)',
        r'\(Acetyl \(K\)\)': '(ac)',
        r'\(GG \(K\)\)': '(ub)',
        r'C\(UniMod\:4\)': 'C',
        r'\[Carbamidomethyl \(C\)\]': '',
        r'M\[Oxidation \(M\)\]': 'M',
        r'\[Acetyl \(Protein N-term\)\]': '(ac)',
        r'\[Phospho \(STY\)\]': '(ph)',
        r'\[GG \(K\)\]': '(ub)',

    }
    for old, new in mappings.items():
        mod_seq = mod_seq.str.replace(old, new, regex=True)
    return mod_seq


def clean_rows(df):
    """
    Cleans unwanted rows such as Contaminants and Decoys.
    If Columns are not present no error is thrown. The index is resetted after this step.

    Parameters
    ----------
    df : pd.DataFrame
        Data with columns that should be removed.

    Returns
    -------
    df : pd.DataFrame
    """
    if 'Contaminant' in df.columns:
        df = df[~df['Contaminant']]
    if 'Decoy' in df.columns:
        df = df[~df['Decoy']]
    df.reset_index(drop=True, inplace=True)
    return df


def aggregate_duplicates(df, keys, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    """
    Aggregates columns such that key columns are unique afterwards. Depending on the column type different aggegation procedures can be performed.

    df : pd.DataFrame
        input data
    keys : array-like
        column names that should be used to create a unique index
    sum_cols : array-like, optional
        column names where duplicated entries will be summed up (cols of dtype numeric)
    first_cols : array-like, optional
        column names where the first row will be used for duplicated entries (cols of dtype categorical)
    max_cols : array-like, optional
        column names where the max value will be used for duplicated entries (cols of dtype numeric)
    min_cols : array-like, optional
        column names where the min value will be used for duplicated entries (cols of dtype numeric)
    concat_cols : array-like, optional
        column names where all values will be concatenated for duplicated entries (cols of dtype object)

    Returns
    -------
    df : pd.DataFrame
    """
    # Double check that expected columns are present
    ui.verify_columns_exist(df, columns=keys + sum_cols + first_cols + max_cols + min_cols + concat_cols)

    # Create a group by objected that will be re-used a few times later
    grouped_df = df.groupby(keys)

    # Count & Sum the duplicates, then merge results to new_df
    grouped_count = grouped_df.size().to_frame().rename({0: 'N duplicates'}, axis=1)
    grouped_sum = grouped_df[sum_cols].sum(min_count=1)
    new_df = pd.merge(left=grouped_count, right=grouped_sum, left_index=True, right_index=True)

    # use the first element of the group
    if first_cols:
        group_first = grouped_df[first_cols].first()
        new_df = pd.merge(left=new_df, right=group_first, left_index=True, right_index=True)

    # use the max value of the group
    if max_cols:
        group_max = grouped_df[max_cols].max()
        new_df = pd.merge(left=new_df, right=group_max, left_index=True, right_index=True)

    # use the min value of the group
    if min_cols:
        group_min = grouped_df[min_cols].min()
        new_df = pd.merge(left=new_df, right=group_min, left_index=True, right_index=True)

    # concatenate all elements of the group
    if concat_cols:
        df[concat_cols] = df[concat_cols].replace(np.nan, '').astype(str)
        for c_col in concat_cols:
            df_concat = grouped_df[c_col].apply(';'.join)
            new_df = pd.merge(left=new_df, right=df_concat, left_index=True, right_index=True)

    # Resort the columns
    df = new_df[['N duplicates'] + first_cols + concat_cols + max_cols + min_cols + sum_cols]
    df.reset_index(inplace=True)
    return df


def assert_constant_within(df, keys, cols):
    """
    Asserts that each group defined by the key columns holds at most one distinct value in cols.

    Some search engines report an already aggregated quantity repeated over the rows of the level
    below it. Such a value must be deduplicated rather than summed or averaged, and this check is
    what makes a misjudged grain fail loudly instead of silently scaling the quantity by the number
    of rows the report happens to carry. Missing values are not counted as a distinct value.

    Parameters
    ----------
    df : pd.DataFrame
        input data
    keys : array-like
        column names defining the groups
    cols : array-like
        column names whose values must be constant within each group

    Returns
    -------
    None
    """
    n_values = df.groupby(keys, dropna=False)[cols].nunique()
    n_violations = int((n_values > 1).any(axis=1).sum())
    if n_violations > 0:
        msg = (f'{n_violations} group(s) defined by {list(keys)} contain more than one distinct value in {list(cols)}. '
               f'The report grain is finer than the level being parsed, or the quantity column is not the aggregated one.')
        raise ValueError(msg)


#
# DIANN
#


def load_diann_lqf_proteins(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = DiannMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.columns = Mapper.rename_general_columns(df.columns)
    df.columns = Mapper.rename_intensity_columns(df.columns)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


def load_diann_lqf_peptide(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = DiannMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.columns = Mapper.rename_general_columns(df.columns)
    df.columns = Mapper.rename_intensity_columns(df.columns)
    df['Modified sequence'] = clean_modified_sequence(df['Modified sequence'])
    df = clean_rows(df)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


#
# MaxQuant
#


def load_mq_tmt_peptides(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = MaxQuantMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.columns = Mapper.rename_general_columns(df.columns)
    df.columns = Mapper.rename_tmt_columns(df.columns)
    df = Mapper.map_indicator_values(df)
    df['Modified sequence'] = clean_modified_sequence(df['Modified sequence'])
    df = clean_rows(df)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


def load_mq_tmt_proteins(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = MaxQuantMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.rename(columns=MaxQuantMap.clean_protein_group_tmt_cols, inplace=True)
    df.columns = Mapper.rename_general_columns(df.columns)
    df.columns = Mapper.rename_tmt_columns(df.columns)
    df = Mapper.map_indicator_values(df)
    if 'Peptides' in df.columns:
        df = df[df['Peptides'] >= 2]
    df = clean_rows(df)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


def load_mq_lfq_peptides(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = MaxQuantMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.columns = Mapper.rename_general_columns(df.columns)
    df = Mapper.map_indicator_values(df)
    df = Mapper.restructure_lfq_evidence(df)
    df.columns = Mapper.rename_intensity_columns(df.columns)
    df['Modified sequence'] = clean_modified_sequence(df['Modified sequence'])
    df = clean_rows(df)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


def load_mq_lqf_proteins(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = MaxQuantMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.columns = Mapper.rename_general_columns(df.columns)
    df.columns = Mapper.rename_lfq_columns(df.columns)
    df = Mapper.map_indicator_values(df)
    if 'Peptides' in df.columns:
        df = df[df['Peptides'] >= 2]
    df = clean_rows(df)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


#
# Proteome Discoverer
#


def load_pd_tmt_peptides(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = PDMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.columns = Mapper.rename_general_columns(df.columns)
    df.columns = Mapper.rename_tmt_columns(df.columns)
    df = Mapper.create_mod_sequence(df)
    df['Modified sequence'] = clean_modified_sequence(df['Modified sequence'])
    df = clean_rows(df)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


def load_pd_tmt_proteins(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    raise NotImplementedError()


def load_pd_lqf_peptides(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    raise NotImplementedError()


def load_pd_lqf_proteins(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = PDMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.columns = Mapper.rename_general_columns(df.columns)
    df.columns = Mapper.rename_intensity_columns(df.columns)
    if 'Peptides' in df.columns:
        df = df[df['Peptides'] >= 2]
    df = clean_rows(df)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


#
# MSFragger
#


def load_fragger_tmt_peptides(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = FraggerMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.columns = Mapper.rename_general_columns(df.columns)
    df.columns = Mapper.rename_tmt_columns(df.columns)
    df = Mapper.create_mod_sequence(df)
    df['Modified sequence'] = clean_modified_sequence(df['Modified sequence'])
    df = clean_rows(df)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


def load_fragger_tmt_proteins(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = FraggerMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.columns = Mapper.rename_general_columns(df.columns)
    df.columns = Mapper.rename_tmt_columns(df.columns)
    if 'Peptides' in df.columns:
        df = df[df['Peptides'] >= 2]
    df = clean_rows(df)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


def load_fragger_lqf_peptides(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = FraggerMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.columns = Mapper.rename_general_columns(df.columns)
    df.columns = Mapper.rename_lfq_columns(df.columns)
    df = Mapper.create_mod_sequence(df)
    df['Modified sequence'] = clean_modified_sequence(df['Modified sequence'])
    df = clean_rows(df)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


def load_fragger_lqf_proteins(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    Mapper = FraggerMap(version)
    df = pd.read_csv(path, sep='\t', low_memory=False)
    df.columns = Mapper.rename_general_columns(df.columns)
    df.columns = Mapper.rename_lfq_columns(df.columns)
    if 'Peptides' in df.columns:
        df = df[df['Peptides'] >= 2]
    df = clean_rows(df)
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df

#
# Spectronaut
#


_SPECTRONAUT_IDENTITY = {
    'PROTEIN': {'Proteins': 'PG.ProteinGroups'},
    'PEPTIDE': {'Modified sequence': 'PEP.GroupingKey', 'Proteins': 'PG.ProteinGroups'},
}

_SPECTRONAUT_QUANTITY = {
    'PROTEIN': 'PG.Quantity',
    'PEPTIDE': 'PEP.Quantity',
}

_SPECTRONAUT_OPTIONAL = {
    'PROTEIN': {'Genes': 'PG.Genes', 'Decoy': 'EG.IsDecoy'},
    'PEPTIDE': {'Genes': 'PG.Genes', 'Decoy': 'EG.IsDecoy', 'Grouping type': 'PEP.GroupingKeyType'},
}

# Spectronaut's quantity is rolled up per run, so the run is what becomes one 'Raw <x>' column.
# R.Label carries the run label the condition setup assigned, and it is the same string a pivot
# report puts in its column headers, so both report shapes name their experiments identically.
_SPECTRONAUT_RUN = 'R.Label'
_SPECTRONAUT_CONDITION = 'R.Condition'

PARQUET_HINT = 'Reading .parquet reports requires pyarrow. Please install it with: pip install curve_curator[parquet]'


def _is_parquet(path):
    return str(path).lower().endswith('.parquet')


def _assert_pyarrow_installed():
    try:
        # Imported here and not at module level: pyarrow is an optional extra, and only a
        # .parquet report needs it.
        import pyarrow
    except ImportError:
        raise ImportError(PARQUET_HINT)


def _is_quantity_of(col, level):
    """
    Checks if a column is the quantity column of a given level, in either dialect and in a long or
    a pivot header. PG.Quantity and PEP.Quantity share the canonical name 'Quantity', so a report
    carrying both must be reduced to the one level before anything is renamed.
    """
    source = _SPECTRONAUT_QUANTITY[level]
    return str(col).endswith(source) or str(col).endswith(_to_parquet_name(source))


def _read_report_header(path):
    """
    Reads only the column names of a report. Both readers are cheap, so the dialect and the report
    shape can be resolved before the body is read with an explicit column list.
    """
    if _is_parquet(path):
        _assert_pyarrow_installed()
        import pyarrow.parquet as pq
        return pd.DataFrame(columns=pq.read_schema(path).names)
    return pd.read_csv(path, sep='\t', nrows=0)


def _read_report(path, columns=None):
    """
    Reads a report, restricted to the given columns. For a precursor-level report this is the
    difference between a few hundred MB and a few MB of pandas memory.
    """
    if _is_parquet(path):
        _assert_pyarrow_installed()
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, sep='\t', usecols=columns, low_memory=False)


def _load_spectronaut(path, version, level, unique_cols, sum_cols, first_cols, max_cols, min_cols, concat_cols):
    """
    Shared body of the two Spectronaut loaders. Reads a long or a pivot report in either the csv or
    the parquet column dialect and returns one row per identity.

    Spectronaut has already aggregated the quantity, so this deduplicates rather than aggregates:
    decoys are dropped first so that the level being parsed sees target evidence only, the quantity
    is asserted constant within its identity, and only then is it collapsed.
    """
    Mapper = SpectronautMap(version)
    identity = _SPECTRONAUT_IDENTITY[level]
    optional = _SPECTRONAUT_OPTIONAL[level]
    other_level = 'PEPTIDE' if level == 'PROTEIN' else 'PROTEIN'

    # Resolve the dialect and the report shape from the header alone. A report commonly carries the
    # roll-up of both levels, and both are called 'Quantity' once renamed, so the other level's
    # column is dropped by its source name first.
    header = _read_report_header(path)
    header = header[[c for c in header.columns if not _is_quantity_of(c, other_level)]]
    source_names = list(header.columns)
    quantity_cols = list(Mapper.rename_quantity_columns(header.columns))
    header.columns = Mapper.rename_general_columns(header.columns)
    canonical = list(header.columns)
    is_long = Mapper.is_long_report(header)
    is_pivot = any(str(c).startswith('Raw ') for c in quantity_cols)

    # A report exported with R.Condition instead of R.Label is the likely mistake, and it is not a
    # column that can stand in: a condition may span several runs, and the quantity differs between
    # them, so keying on it would ask the parser to collapse measurements that are not duplicates.
    if not is_long and not is_pivot and _SPECTRONAUT_CONDITION in source_names:
        raise ValueError(
            f'The Spectronaut report has a "{_SPECTRONAUT_CONDITION}" column but no "{_SPECTRONAUT_RUN}" column. '
            f'CurveCurator identifies an experiment by its run, because the quantity is rolled up per run and one '
            f'condition may cover several runs. Please add "{_SPECTRONAUT_RUN}" to the report schema in Spectronaut.')

    if not is_long and not is_pivot:
        raise ValueError(
            'The Spectronaut report is neither a long report (no "R.Label" column) nor a pivot report '
            '(no "<run>.PG.Quantity" or "<run>.PEP.Quantity" columns). '
            f'The report contains: {source_names}.')

    required = dict(identity)
    if is_long:
        required['Quantity'] = _SPECTRONAUT_QUANTITY[level]
    for name, source in required.items():
        if name not in canonical:
            raise ValueError(f'The Spectronaut report has no "{name}" column. '
                             f'Please add "{source}" to the report schema in Spectronaut.')

    # Read only the columns that are actually needed.
    wanted = {'Run'} | set(required) | set(optional)
    source_cols = [c for c, name in zip(source_names, canonical) if name in wanted]
    if not is_long:
        source_cols += [c for c, name in zip(source_names, quantity_cols) if str(name).startswith('Raw ')]

    df = _read_report(path, columns=source_cols)
    df.columns = Mapper.rename_general_columns(df.columns)
    if not is_long:
        df.columns = Mapper.rename_quantity_columns(df.columns)
    df = Mapper.map_indicator_values(df)

    # Decoys go before any quantity is read. EG.IsDecoy flags an elution group, while the quantity
    # is an aggregate over target evidence, so a decoy row left in place would make a consistent
    # protein group fail the constancy check below.
    df = clean_rows(df)

    # Guarantee the annotation column exists, so first_cols can name it unconditionally.
    if 'Genes' not in df.columns:
        df['Genes'] = df['Proteins']

    if 'Grouping type' in df.columns:
        grains = sorted(set(df['Grouping type'].dropna()))
        if grains:
            ui.message(f" * Spectronaut grouped the peptides by: {', '.join(grains)}.")

    if is_long:
        if df['Run'].isna().any():
            raise ValueError(f'The "{_SPECTRONAUT_RUN}" column contains empty values. Please give every run a label '
                             'in the Spectronaut condition setup and export the report again.')
        assert_constant_within(df, keys=unique_cols + ['Run'], cols=['Quantity'])
        df = Mapper.restructure_long_report(df, index=unique_cols, value_col='Quantity')
    else:
        assert_constant_within(df, keys=unique_cols, cols=[c for c in df.columns if str(c).startswith('Raw ')])

    if level == 'PEPTIDE':
        df['Modified sequence'] = clean_modified_sequence(df['Modified sequence'])

    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols,
                              min_cols=min_cols, concat_cols=concat_cols)
    return df


def load_spectronaut_dia_proteins(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    return _load_spectronaut(path, version, 'PROTEIN', unique_cols, sum_cols, first_cols, max_cols, min_cols, concat_cols)


def load_spectronaut_dia_peptides(path, version, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    return _load_spectronaut(path, version, 'PEPTIDE', unique_cols, sum_cols, first_cols, max_cols, min_cols, concat_cols)


#
# Generic
#

def test_missing_columns(df, cols):
    for col in cols:
        if col not in df.columns:
            ui.error(f'Column <{col}> was not found in the input file.')
            raise ValueError(f'The input file must contain a <{cols}> column. Please add to the input file.')

def remove_missing_column(df, cols):
    cleaned_cols = []
    for col in cols:
        if col in df.columns:
            cleaned_cols.append(col)
        else:
            ui.warning(f'Column <{col}> was not found in the input file. Continue anyways.')
    return cleaned_cols


def load_generic(path, unique_cols, sum_cols=[], first_cols=[], max_cols=[], min_cols=[], concat_cols=[]):
    # Load and test
    df = pd.read_csv(path, sep='\t', low_memory=False)
    test_missing_columns(df, unique_cols) # A unique column is a requirement for generic upload

    # Make robust against missingness
    sum_cols = remove_missing_column(df, sum_cols)
    first_cols = remove_missing_column(df, first_cols)
    max_cols = remove_missing_column(df, max_cols)
    min_cols = remove_missing_column(df, min_cols)
    concat_cols = remove_missing_column(df, concat_cols)

    # aggregate
    df = aggregate_duplicates(df, keys=unique_cols, sum_cols=sum_cols, first_cols=first_cols, max_cols=max_cols, min_cols=min_cols,
                              concat_cols=concat_cols)
    return df


def load(config):
    """
    Load the input data. Depending on the particular data type use different parser functions.
    The different parsers will return a unified data table for downstream analysis.
    """
    # toml parameters
    path = config['Paths'].get('input_file')
    measurement_type = config['Experiment'].get('measurement_type', 'OTHER').upper()  # <LFQ|TMT|DIA|OTHER>
    data_type = config['Experiment'].get('data_type', 'OTHER').upper()  # <PEPTIDE|PROTEIN|OTHER>
    search_engine = config['Experiment'].get('search_engine', 'OTHER').upper()  # <MAXQUANT|DIANN|OTHER>
    search_engine_version = config['Experiment'].get('search_engine_version', '0.0.0')  # search engine version
    experiments = config['Experiment'].get('experiments')
    ui.message(f' * Loading data file: {path}.')
    ui.message(f' * Parser mode: ({data_type}, {measurement_type}, {search_engine}).')

    # columns
    raw_cols = [f'Raw {e}' for e in experiments]

    if (measurement_type == 'DIA') and (search_engine == 'DIANN') and (data_type == 'PROTEIN'):
        unique_cols = ['Genes']
        df = load_diann_lqf_proteins(path, search_engine_version, unique_cols=unique_cols, sum_cols=raw_cols)
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']

    elif (measurement_type == 'DIA') and (search_engine == 'DIANN') and (data_type == 'PEPTIDE'):
        # TODO: make this to dictionary
        unique_cols = ['Modified sequence']
        first_cols = ['Genes', 'Proteins']
        df = load_diann_lqf_peptide(path, search_engine_version, unique_cols=unique_cols, first_cols=first_cols, sum_cols=raw_cols)
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']

    elif (measurement_type == 'DIA') and (search_engine == 'SPECTRONAUT') and (data_type == 'PROTEIN'):
        unique_cols = ['Proteins']
        first_cols = ['Genes']
        # max_cols, not sum_cols: Spectronaut reports an already aggregated quantity, and the loader
        # has asserted it is constant within the protein group, so the max is that value.
        df = load_spectronaut_dia_proteins(path, search_engine_version, unique_cols=unique_cols, first_cols=first_cols,
                                           max_cols=raw_cols)
        if 'Genes' not in df.columns:
            df['Genes'] = df['Proteins']
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']

    elif (measurement_type == 'DIA') and (search_engine == 'SPECTRONAUT') and (data_type == 'PEPTIDE'):
        unique_cols = ['Modified sequence']
        first_cols = ['Proteins', 'Genes']
        # max_cols, not sum_cols: see the protein branch above.
        df = load_spectronaut_dia_peptides(path, search_engine_version, unique_cols=unique_cols, first_cols=first_cols,
                                           max_cols=raw_cols)
        if 'Genes' not in df.columns:
            df['Genes'] = df['Proteins']
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']

    elif (measurement_type == 'LFQ') and (search_engine == 'MAXQUANT') and (data_type == 'PROTEIN'):
        # TODO: make this to dictionary
        unique_cols = ['Genes', 'Proteins']
        max_cols = ['Score']
        sum_cols = raw_cols + ['Peptides']
        df = load_mq_lqf_proteins(path, search_engine_version, unique_cols=unique_cols, sum_cols=sum_cols, max_cols=max_cols)
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']

    elif (measurement_type == 'LFQ') and (search_engine == 'PD') and (data_type == 'PROTEIN'):
        # TODO: make this to dictionary
        unique_cols = ['Proteins']
        max_cols = ['Score']
        sum_cols = raw_cols + ['Peptides']
        df = load_pd_lqf_proteins(path, search_engine_version, unique_cols=unique_cols, sum_cols=sum_cols, max_cols=max_cols)
        if 'Genes' not in df.columns:
            df['Genes'] = df['Proteins']
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']

    elif (measurement_type == 'LFQ') and (search_engine == 'MSFRAGGER') and (data_type == 'PROTEIN'):
        unique_cols = ['Proteins', 'Genes']
        sum_cols = raw_cols + ['Peptides']
        df = load_fragger_lqf_proteins(path, search_engine_version, unique_cols=unique_cols, sum_cols=sum_cols)
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']

    elif (measurement_type == 'LFQ') and (search_engine == 'MAXQUANT') and (data_type == 'PEPTIDE'):
        # TODO: make this to dictionary
        unique_cols = ['Modified sequence']
        first_cols = ['Genes', 'Proteins']
        max_cols = ['Score']
        df = load_mq_lfq_peptides(path, search_engine_version, unique_cols=unique_cols, first_cols=first_cols, sum_cols=raw_cols, max_cols=max_cols)
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']

    elif (measurement_type == 'LFQ') and (search_engine == 'MSFRAGGER') and (data_type == 'PEPTIDE'):
        # TODO: make this to dictionary
        unique_cols = ['Modified sequence']
        first_cols = ['Proteins', 'Genes']
        max_cols = []
        df = load_fragger_lqf_peptides(path, search_engine_version, unique_cols=unique_cols, sum_cols=raw_cols, first_cols=first_cols, max_cols=max_cols)
        if 'Genes' not in df.columns:
            df['Genes'] = df['Proteins']
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']

    elif (measurement_type == 'TMT') and (search_engine == 'MAXQUANT') and (data_type == 'PROTEIN'):
        unique_cols = ['Proteins', 'Genes']
        max_cols = ['Score']
        sum_cols = raw_cols + ['Peptides']
        df = load_mq_tmt_proteins(path, search_engine_version, unique_cols=unique_cols, sum_cols=sum_cols, max_cols=max_cols)
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']

    elif (measurement_type == 'TMT') and (search_engine == 'MSFRAGGER') and (data_type == 'PROTEIN'):
        unique_cols = ['Proteins', 'Genes']
        sum_cols = raw_cols + ['Peptides']
        df = load_fragger_tmt_proteins(path, search_engine_version, unique_cols=unique_cols, sum_cols=sum_cols)
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']

    elif (measurement_type == 'TMT') and (search_engine == 'MAXQUANT') and (data_type == 'PEPTIDE'):
        # TODO: make this to dictionary
        unique_cols = ['Modified sequence']
        first_cols = ['Genes', 'Proteins']
        max_cols = ['Score']
        df = load_mq_tmt_peptides(path, search_engine_version, unique_cols=unique_cols, sum_cols=raw_cols, first_cols=first_cols, max_cols=max_cols)
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']

    elif (measurement_type == 'TMT') and (search_engine == 'PD') and (data_type == 'PEPTIDE'):
        # TODO: make this to dictionary
        unique_cols = ['Modified sequence']
        first_cols = ['Proteins']
        max_cols = []
        df = load_pd_tmt_peptides(path, search_engine_version, unique_cols=unique_cols, sum_cols=raw_cols, first_cols=first_cols, max_cols=max_cols)
        if 'Genes' not in df.columns:
            df['Genes'] = df['Proteins']
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']

    elif (measurement_type == 'TMT') and (search_engine == 'MSFRAGGER') and (data_type == 'PEPTIDE'):
        # TODO: make this to dictionary
        unique_cols = ['Modified sequence']
        first_cols = ['Proteins', 'Genes']
        max_cols = []
        df = load_fragger_tmt_peptides(path, search_engine_version, unique_cols=unique_cols, sum_cols=raw_cols, first_cols=first_cols, max_cols=max_cols)
        if 'Genes' not in df.columns:
            df['Genes'] = df['Proteins']
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']

    elif (measurement_type == 'TMT') and (search_engine == 'OTHER') and (data_type == 'PEPTIDE'):
        df = load_generic_peptide_format(path)
        if 'Name' not in df.columns:
            if 'Genes' in df.columns:
                df['Name'] = df['Genes']
            else:
                df['Name'] = df.index.values.copy()

    elif (measurement_type == 'TMT') and (search_engine == 'OTHER') and (data_type == 'PROTEIN'):
        df = load_generic_protein_format(path)
        if 'Name' not in df.columns:
            if 'Genes' in df.columns:
                df['Name'] = df['Genes']
            else:
                df['Name'] = df.index.values.copy()

    elif (measurement_type == 'OTHER') and (search_engine == 'OTHER') and (data_type == 'PEPTIDE'):
        unique_cols = ['Modified sequence']
        first_cols = ['Genes', 'Proteins']
        max_cols = ['Score']
        df = load_generic(path, unique_cols=unique_cols, sum_cols=raw_cols, first_cols=first_cols, max_cols=max_cols)
        if 'Name' not in df.columns:
            if 'Genes' in df.columns:
                df['Name'] = df['Genes']
            else:
                df['Name'] = df['Modified sequence']

    elif (measurement_type == 'OTHER') and (search_engine == 'OTHER') and (data_type == 'PROTEIN'):
        unique_cols = ['Genes', 'Proteins']
        max_cols = ['Score']
        sum_cols = raw_cols + ['Peptides']
        df = load_generic(path, unique_cols=unique_cols, sum_cols=sum_cols, max_cols=max_cols)
        if 'Name' not in df.columns:
            df['Name'] = df['Genes']

    elif (measurement_type == 'OTHER') and (search_engine == 'OTHER') and (data_type == 'OTHER'):
        unique_cols = ['Name']
        df = load_generic(path, unique_cols=unique_cols, sum_cols=raw_cols)

    else:
        msg = f'The combination of measurement_type = "{measurement_type}", data type = "{data_type}", and  search_engine = "{search_engine}" is currently not supported.'
        raise NotImplementedError(msg)

    # if there are completely empty columns, drop empty columns from the input df and remove it from the config file
    empty_cols = (df[raw_cols].isna().mean() == 1.0)
    empty_cols = empty_cols[empty_cols].index
    empty_experiments = set(empty_cols.str.replace('Raw ', ''))
    if len(empty_experiments) > 0:
        df = df.drop(empty_cols, axis='columns')
        keep_idx = [i for i, e in enumerate(config['Experiment']['experiments']) if e not in empty_experiments]
        config['Experiment']['experiments'] = config['Experiment']['experiments'][keep_idx].copy()
        config['Experiment']['doses'] = config['Experiment']['doses'][keep_idx].copy()
        config['Experiment']['control_experiment'] = np.array([e for e in config['Experiment']['control_experiment'] if e not in empty_experiments])
        ui.warning(' * The following experiments have no values in the data file and have been removed from the analysis:', end='\n')
        ui.warning('   {}'.format(sorted(empty_experiments)))

    return df
