"""Deterministic synthetic workflow; run as a module from repository root."""
import argparse
import copy
import json
from pathlib import Path
import platform
import runpy
from importlib.metadata import version

import numpy as np
import pandas as pd
from curve_curator import data_parser, toml_parser, quantification, thresholding, dashboard, data_simulator


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    doses = [0., 1., 3., 10., 30., 100., 300., 1000., 3000.]
    runs = [f'dose_{i}' for i in range(len(doses))]
    rng = np.random.default_rng(20261106)
    records = []
    for i in range(24):
        back = [.15, 1., 2.5][i % 3]
        ratios = back + (1 - back) / (1 + np.array(doses) / (30 + i))
        quantities = (10000 + i * 1000) * ratios * rng.uniform(.97, 1.03, len(doses))
        if i % 4 == 0:
            quantities[3] = np.nan
        if i == 23:
            quantities[1:] = np.nan  # must be filtered
        for j, run in enumerate(runs):
            row = {'R.Label': run, 'PG.ProteinGroups': f'P{i:03}',
                   'PG.Genes': f'G{i:03}' if i % 5 else np.nan,
                   'PG.Quantity': quantities[j], 'PEP.Quantity': 999999.}
            records.extend([row, row.copy()])  # repeated precursor rows must not sum PG.Quantity
    source = output / 'synthetic.tsv'
    pd.DataFrame(records).to_csv(source, sep='\t', index=False)
    config = toml_parser.set_default_values({
        '__file__': {'Path': str(output / 'parameters.toml')},
        'Meta': {'id': 'synthetic', 'condition': 'synthetic', 'description': 'Synthetic pandas compatibility'},
        'Paths': {'input_file': str(source), 'curves_file': str(output / 'curves.tsv'),
                  'dashboard': str(output / 'dashboard.html')},
        'Experiment': {'experiments': np.array(runs), 'doses': np.array(doses), 'dose_scale': 1e-9,
                       'dose_unit': 'M', 'control_experiment': np.array([runs[0]]),
                       'measurement_type': 'DIA', 'data_type': 'PROTEIN',
                       'search_engine': 'SPECTRONAUT', 'search_engine_version': '19.9'},
        'Processing': {'available_cores': 1, 'max_missing': 2, 'max_imputation': 2,
                       'imputation': True, 'normalization': True},
        'Curve Fit': {}, 'F Statistic': {'alpha': .05, 'fc_lim': .5}, 'Dashboard': {},
    })
    data = data_parser.load(config)
    data.to_csv(output / 'parsed.tsv', sep='\t', index=False)
    # Snapshot the existing peptide and native PTM fixtures against the same
    # expected parser contracts covered by unit tests in every environment.
    peptide = runpy.run_path('tests/unit/test_parser_spectronaut.py')
    peptide_path = peptide['write_long_tsv'](output / 'peptide-input.txt')
    peptide_config = copy.deepcopy(config)
    peptide_config['Paths']['input_file'] = peptide_path
    peptide_config['Experiment']['data_type'] = 'PEPTIDE'
    peptide_config['Experiment']['experiments'] = np.array(['dose_0', 'dose_1'])
    data_parser.load(peptide_config).to_csv(output / 'parsed_peptides.tsv', sep='\t', index=False)
    ptm = runpy.run_path('tests/unit/test_parser_spectronaut_ptm.py')
    for i, quantity in enumerate(['PTM.Quantity', 'PTM.QuantityPerProtein', 'PTM.Stoichiometry']):
        folder = output / f'ptm-{i}'
        folder.mkdir(exist_ok=True)
        path = ptm['write_report'](folder, ptm['report']())
        data_parser.load(ptm['config'](path, quantity)).to_csv(output / f'parsed_ptm_{i}.tsv', sep='\t', index=False)
    data = quantification.run_pipeline(data, config)
    sam = thresholding.apply_significance_thresholds(data.copy(), config)
    sam.to_csv(output / 'sam.tsv', sep='\t', index=False)
    bh_config = copy.deepcopy(config)
    bh_config['F Statistic']['mtc_method'] = 'fdr_bh'
    bh = thresholding.apply_significance_thresholds(data.copy(), bh_config)
    bh.to_csv(output / 'bh.tsv', sep='\t', index=False)
    np.random.seed(20261106)
    decoys = data_simulator.get_decoys(sam, config)
    decoys.to_csv(output / 'decoy_inputs.tsv', sep='\t', index=False)
    # Empirical noise differs at floating point roundoff between pandas versions.
    # Quantize synthetic decoy inputs to 12 decimals so ill-conditioned null fits
    # receive identical numerical inputs. Production simulation is unchanged.
    raw = [f'Raw {run}' for run in runs]
    decoys[raw] = decoys[raw].round(12)
    decoys = quantification.run_pipeline(decoys, config, decoy_mode=True)
    decoys = thresholding.apply_significance_thresholds(decoys, config)
    thresholding.estimate_qvalues(sam, decoys, config)
    sam.to_csv(output / 'qvalues.tsv', sep='\t', index=False)
    decoys.to_csv(output / 'decoys.tsv', sep='\t', index=False)
    dashboard.render(sam.copy(), config)
    html = (output / 'dashboard.html').read_text()
    assert 'Bokeh' in html and len(html) > 10000
    assert len(sam) == 15  # missingness and below-imputation-floor filters
    (output / 'versions.json').write_text(json.dumps({
        'python': platform.python_version(), **{p: version(p) for p in
        ['pandas', 'numpy', 'scipy', 'bokeh', 'statsmodels', 'pyarrow', 'pytest']}}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    run(parser.parse_args().output)
