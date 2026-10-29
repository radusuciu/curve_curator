import numpy as np
import pandas as pd
import pytest

from curve_curator.data_parser import load
from curve_curator.toml_parser import check_for_unknown_keys


def config(path, quantity='PTM.Quantity'):
    return {
        'Paths': {'input_file': path},
        'Experiment': {
            'data_type': 'PEPTIDE', 'measurement_type': 'DIA', 'search_engine': 'SPECTRONAUT',
            'search_engine_version': '19.9', 'quantity_column': quantity,
            'experiments': np.array(['Run A.1', 'Run B.2']), 'doses': np.array([0, 1]),
            'control_experiment': np.array(['Run A.1']),
        },
    }


def report():
    return pd.DataFrame({
        'PTM.CollapseKey': ['P1_S10_[Phospho]_1'] * 4 + ['P1_S10_[Phospho]_2'] * 2,
        'R.Label': ['Run A.1', 'Run A.1', 'Run B.2', 'Run B.2', 'Run A.1', 'Run B.2'],
        'PTM.Quantity': [10., 10., 20., 20., 30., 40.],
        'PTM.QuantityPerProtein': [1., 1., 2., 2., 3., 4.],
        'PTM.Stoichiometry': [.1, .1, .2, .2, .3, .4],
        'PEP.Quantity': [999.] * 6, 'PG.Quantity': [9999.] * 6,
        'PTM.ProteinId': ['P1'] * 6, 'PG.Genes': ['G1'] * 6,
        'PTM.SiteLocation': [10] * 6, 'PTM.SiteAA': ['S'] * 6,
        'PTM.ModificationTitle': ['Phospho (STY)'] * 6,
        'PTM.Multiplicity': [1] * 4 + [2] * 2,
        'EG.ModifiedSequence': ['_ZpSK_', '_ApSK_', '_ZpSK_', '_ApSK_', '_DOUBLE_', '_DOUBLE_'],
        'EG.IsDecoy': [False] * 6,
    })


def write_report(tmp_path, df, pivot=False, parquet=False):
    if pivot:
        identity = ['PTM.CollapseKey']
        annotations = df.drop(columns=['R.Label', 'PTM.Quantity', 'PTM.QuantityPerProtein',
                                       'PTM.Stoichiometry', 'PG.Quantity', 'PEP.Quantity'])
        annotations = annotations.drop_duplicates('PTM.CollapseKey')
        wide = annotations.set_index('PTM.CollapseKey')
        for quantity in ['PTM.Quantity', 'PTM.QuantityPerProtein', 'PTM.Stoichiometry', 'PG.Quantity', 'PEP.Quantity']:
            values = df.groupby(identity + ['R.Label'])[quantity].max().unstack()
            values.columns = [f'[{i}] {run}.{quantity}' for i, run in enumerate(values.columns, 1)]
            wide = wide.join(values)
        df = wide.reset_index()
    path = tmp_path / ('report.parquet' if parquet else 'report.tsv')
    if parquet:
        pytest.importorskip('pyarrow')
        df = df.rename(columns=lambda col: col.replace('.', '_').replace(' ', '_'))
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, sep='\t', index=False)
    return path


@pytest.mark.parametrize('pivot', [False, True], ids=['long', 'pivot'])
@pytest.mark.parametrize('parquet', [False, True], ids=['tsv', 'parquet'])
@pytest.mark.parametrize('quantity,scale', [('PTM.Quantity', 1), ('PTM.QuantityPerProtein', .1),
                                          ('PTM.Stoichiometry', .01)])
def test_quantity_selection_and_site_identity(tmp_path, pivot, parquet, quantity, scale):
    path = write_report(tmp_path, report(), pivot, parquet)
    result = load(config(path, quantity))
    assert result['Grouping key'].tolist() == ['P1_S10_[Phospho]_1', 'P1_S10_[Phospho]_2']
    assert result['Name'].equals(result['Grouping key'])
    assert result['Grouping type'].tolist() == ['PTM Site'] * 2
    np.testing.assert_allclose(result[['Raw Run A.1', 'Raw Run B.2']], np.array([[10, 20], [30, 40]]) * scale)
    assert result['Multiplicity'].tolist() == ['1', '2']


def test_sequences_aggregate_without_changing_identity(tmp_path):
    result = load(config(write_report(tmp_path, report())))
    assert result['Modified sequence'].iloc[0] == '_ApSK_;_ZpSK_'


@pytest.mark.parametrize('column', ['PTM.Quantity', 'PTM.ProteinId', 'PTM.SiteLocation',
                                    'PTM.SiteAA', 'PTM.Multiplicity', 'PTM.ModificationTitle'])
def test_conflicting_measurements_or_site_annotations_raise(tmp_path, column):
    df = report()
    df.loc[1, column] = 11. if column in ['PTM.Quantity', 'PTM.SiteLocation', 'PTM.Multiplicity'] else 'other'
    with pytest.raises(ValueError, match='more than one distinct value'):
        load(config(write_report(tmp_path, df)))


def test_decoys_removed_before_validation(tmp_path):
    df = report()
    df.loc[1, 'EG.IsDecoy'] = True
    df.loc[1, 'PTM.Quantity'] = 999.
    result = load(config(write_report(tmp_path, df)))
    assert result['Raw Run A.1'].tolist() == [10., 30.]


@pytest.mark.parametrize('column', ['PTM.CollapseKey', 'R.Label'])
@pytest.mark.parametrize('value', [None, '', '   '])
def test_empty_identities_raise(tmp_path, column, value):
    df = report()
    df.loc[0, column] = value
    with pytest.raises(ValueError, match='empty'):
        load(config(write_report(tmp_path, df)))


@pytest.mark.parametrize('column', ['PTM.CollapseKey', 'PTM.Quantity', 'R.Label'])
def test_required_columns_raise(tmp_path, column):
    df = report().drop(columns=column)
    with pytest.raises(ValueError, match='requires'):
        load(config(write_report(tmp_path, df)))


def test_optional_annotations_not_required(tmp_path):
    df = report()[['PTM.CollapseKey', 'R.Label', 'PTM.Quantity']]
    result = load(config(write_report(tmp_path, df)))
    assert len(result) == 2
    assert result['Name'].equals(result['Grouping key'])


def test_repeated_missing_quantity_preserves_measurement(tmp_path):
    df = report()
    df.loc[0, 'PTM.Quantity'] = np.nan
    result = load(config(write_report(tmp_path, df)))
    assert result['Raw Run A.1'].tolist() == [10., 30.]


@pytest.mark.parametrize('conflict', [False, True])
def test_pivot_duplicate_rows_are_validated_before_deduplication(tmp_path, conflict):
    df = pd.DataFrame({
        'PTM.CollapseKey': ['P1_S10_1'] * 2,
        '[1] Run A.1.PTM.Quantity': [0., 1. if conflict else 0.],
        '[2] Run B.2.PTM.Quantity': [np.nan, 20.],
        'EG.ModifiedSequence': ['_ZpSK_', '_ApSK_'],
    })
    path = tmp_path / 'pivot.tsv'
    df.to_csv(path, sep='\t', index=False)
    if conflict:
        with pytest.raises(ValueError, match='more than one distinct value'):
            load(config(path))
    else:
        result = load(config(path))
        assert result[['Raw Run A.1', 'Raw Run B.2']].values.tolist() == [[0., 20.]]
        assert result['Modified sequence'].tolist() == ['_ApSK_;_ZpSK_']


@pytest.mark.parametrize('parquet', [False, True])
def test_missing_site_measurement_remains_missing(tmp_path, parquet):
    df = report().drop(index=5)
    result = load(config(write_report(tmp_path, df, parquet=parquet)))
    assert np.isnan(result['Raw Run B.2'].iloc[1])


def test_ambiguous_parquet_run_names_raise(tmp_path):
    path = write_report(tmp_path, report(), pivot=True, parquet=True)
    cfg = config(path)
    cfg['Experiment']['experiments'] = ['Run A.1', 'Run_A_1']
    with pytest.raises(ValueError, match='ambiguous'):
        load(cfg)


@pytest.mark.parametrize('pivot', [False, True])
def test_missing_configured_run_raises(tmp_path, pivot):
    path = write_report(tmp_path, report(), pivot=pivot)
    cfg = config(path)
    cfg['Experiment']['experiments'] = ['missing']
    with pytest.raises(ValueError, match='missing configured runs'):
        load(cfg)


@pytest.mark.parametrize('quantity', ['PTM.Unknown', 'PG.Quantity', ''])
def test_unknown_quantity_rejected(tmp_path, quantity):
    with pytest.raises(ValueError, match='Unsupported Spectronaut PTM quantity'):
        load(config(tmp_path / 'unused.tsv', quantity))


def test_quantity_option_requires_spectronaut_peptide(tmp_path):
    cfg = config(tmp_path / 'unused.tsv')
    cfg['Experiment']['data_type'] = 'PROTEIN'
    with pytest.raises(ValueError, match='PEPTIDE imports only'):
        load(cfg)


def test_quantity_option_registered_in_toml(tmp_path, capsys):
    check_for_unknown_keys('Experiment', config(tmp_path / 'unused.tsv'))
    assert 'quantity_column' not in capsys.readouterr().out


@pytest.mark.parametrize('pivot', [False, True])
def test_unconfigured_runs_are_not_returned(tmp_path, pivot):
    path = write_report(tmp_path, report(), pivot=pivot)
    cfg = config(path)
    cfg['Experiment']['experiments'] = ['Run A.1']
    result = load(cfg)
    assert result['Raw Run A.1'].tolist() == [10., 30.]
    assert 'Raw Run B.2' not in result.columns


@pytest.mark.parametrize('data_type,quantity,expected', [('PROTEIN', 'PG.Quantity', 9999.),
                                                       ('PEPTIDE', 'PEP.Quantity', 999.)])
@pytest.mark.parametrize('pivot', [False, True])
def test_other_rollups_ignore_ptm_columns(tmp_path, data_type, quantity, expected, pivot):
    df = report()
    df['PG.ProteinGroups'] = 'P1'
    df['PEP.GroupingKey'] = '_PEPTIDEK_'
    if pivot:
        df = pd.DataFrame({
            'PG.ProteinGroups': ['P1'], 'PEP.GroupingKey': ['_PEPTIDEK_'],
            'PTM.CollapseKey': ['P1_S10_[Phospho]_1'],
            **{f'{run}.{response}': [value]
               for run in ['Run A.1', 'Run B.2']
               for response, value in [(quantity, expected), ('PTM.Quantity', 10.)]},
        })
    path = tmp_path / 'mixed.tsv'
    df.to_csv(path, sep='\t', index=False)
    cfg = config(path)
    cfg['Experiment'].pop('quantity_column')
    cfg['Experiment']['data_type'] = data_type
    result = load(cfg)
    assert result['Raw Run A.1'].tolist() == [expected]
    assert result['Raw Run B.2'].tolist() == [expected]
    assert 'Grouping key' not in result.columns
