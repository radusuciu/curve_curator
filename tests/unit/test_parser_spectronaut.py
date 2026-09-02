import pandas as pd
import numpy as np
import pytest

from curve_curator.search_engine_outputs.Spectronaut import SpectronautMap
from curve_curator.data_parser import assert_constant_within, load_spectronaut_dia_proteins, load_spectronaut_dia_peptides


class TestSpectronautMap:
    def test_rename_general_columns_csv_dialect(self):
        cols = pd.Index(['R.Condition', 'PG.ProteinGroups', 'PG.Genes', 'PEP.GroupingKey', 'PEP.GroupingKeyType', 'EG.IsDecoy', 'PG.Quantity'])
        expected_result = pd.Index(['Condition', 'Proteins', 'Genes', 'Modified sequence', 'Grouping type', 'Decoy', 'Quantity'])
        assert SpectronautMap.rename_general_columns(cols).equals(expected_result)

    def test_rename_general_columns_parquet_dialect(self):
        cols = pd.Index(['R_Condition', 'PG_ProteinGroups', 'PG_Genes', 'PEP_GroupingKey', 'PEP_GroupingKeyType', 'EG_IsDecoy', 'PEP_Quantity'])
        expected_result = pd.Index(['Condition', 'Proteins', 'Genes', 'Modified sequence', 'Grouping type', 'Decoy', 'Quantity'])
        assert SpectronautMap.rename_general_columns(cols).equals(expected_result)

    def test_rename_general_columns_unknown_columns(self):
        cols = pd.Index(['FG.Charge', 'EG.PrecursorId', 'Something_Else'])
        expected_result = pd.Index(['FG.Charge', 'EG.PrecursorId', 'Something_Else'])
        assert SpectronautMap.rename_general_columns(cols).equals(expected_result)

    def test_map_indicator_values_strings(self):
        df = pd.DataFrame({'Decoy': ['True', 'False', 'False', 'True']})
        expected_result = pd.DataFrame({'Decoy': [True, False, False, True]})
        assert SpectronautMap.map_indicator_values(df).equals(expected_result)

    def test_map_indicator_values_booleans(self):
        df = pd.DataFrame({'Decoy': [True, False, np.nan]})
        expected_result = pd.DataFrame({'Decoy': [True, False, False]})
        assert SpectronautMap.map_indicator_values(df).equals(expected_result)

    def test_map_indicator_values_absent_column(self):
        df = pd.DataFrame({'Proteins': ['P1', 'P2']})
        expected_result = pd.DataFrame({'Proteins': ['P1', 'P2']})
        assert SpectronautMap.map_indicator_values(df).equals(expected_result)


class TestRenameQuantityColumns:
    def test_indexed_pivot_header(self):
        cols = pd.Index(['PG.ProteinGroups', '[1] CondA.PG.Quantity', '[2] CondB.PG.Quantity'])
        expected_result = pd.Index(['PG.ProteinGroups', 'Raw CondA', 'Raw CondB'])
        assert SpectronautMap.rename_quantity_columns(cols).equals(expected_result)

    def test_pivot_header_without_index_prefix(self):
        cols = pd.Index(['CondA.PG.Quantity', 'CondB.PG.Quantity'])
        expected_result = pd.Index(['Raw CondA', 'Raw CondB'])
        assert SpectronautMap.rename_quantity_columns(cols).equals(expected_result)

    def test_peptide_pivot_header(self):
        cols = pd.Index(['PEP.GroupingKey', '[10] Cond A.PEP.Quantity'])
        expected_result = pd.Index(['PEP.GroupingKey', 'Raw Cond A'])
        assert SpectronautMap.rename_quantity_columns(cols).equals(expected_result)

    def test_unrelated_columns_pass_through(self):
        cols = pd.Index(['PG.Quantity', 'PEP.GroupingKey', 'R.Condition'])
        expected_result = pd.Index(['PG.Quantity', 'PEP.GroupingKey', 'R.Condition'])
        assert SpectronautMap.rename_quantity_columns(cols).equals(expected_result)


class TestRestructureLongReport:
    def test_long_to_wide(self):
        df = pd.DataFrame({
            'Proteins': ['P1', 'P1', 'P2', 'P2'],
            'Genes': ['G1', 'G1', 'G2', 'G2'],
            'Condition': ['C1', 'C2', 'C1', 'C2'],
            'Quantity': [10.0, 20.0, 30.0, 40.0],
        })
        expected_result = pd.DataFrame({
            'Proteins': ['P1', 'P2'],
            'Raw C1': [10.0, 30.0],
            'Raw C2': [20.0, 40.0],
            'Genes': ['G1', 'G2'],
        })
        result = SpectronautMap.restructure_long_report(df, index=['Proteins'], value_col='Quantity')
        assert result.equals(expected_result)

    def test_missing_condition_becomes_nan(self):
        df = pd.DataFrame({
            'Proteins': ['P1', 'P1', 'P2'],
            'Genes': ['G1', 'G1', 'G2'],
            'Condition': ['C1', 'C2', 'C1'],
            'Quantity': [10.0, 20.0, 30.0],
        })
        expected_result = pd.DataFrame({
            'Proteins': ['P1', 'P2'],
            'Raw C1': [10.0, 30.0],
            'Raw C2': [20.0, np.nan],
            'Genes': ['G1', 'G2'],
        })
        result = SpectronautMap.restructure_long_report(df, index=['Proteins'], value_col='Quantity')
        assert result.equals(expected_result)

    def test_empty_annotation_is_not_dropped(self):
        df = pd.DataFrame({
            'Proteins': ['P1', 'P1', 'P2', 'P2'],
            'Genes': ['G1', 'G1', np.nan, np.nan],
            'Condition': ['C1', 'C2', 'C1', 'C2'],
            'Quantity': [10.0, 20.0, 30.0, 40.0],
        })
        expected_result = pd.DataFrame({
            'Proteins': ['P1', 'P2'],
            'Raw C1': [10.0, 30.0],
            'Raw C2': [20.0, 40.0],
            'Genes': ['G1', np.nan],
        })
        result = SpectronautMap.restructure_long_report(df, index=['Proteins'], value_col='Quantity')
        assert result.equals(expected_result)


class TestAssertConstantWithin:
    def test_constant_values_pass(self):
        df = pd.DataFrame({
            'Proteins': ['P1', 'P1', 'P2', 'P2'],
            'Condition': ['C1', 'C1', 'C1', 'C2'],
            'Quantity': [10.0, 10.0, 30.0, 40.0],
        })
        assert assert_constant_within(df, keys=['Proteins', 'Condition'], cols=['Quantity']) is None

    def test_missing_values_do_not_count(self):
        df = pd.DataFrame({
            'Proteins': ['P1', 'P1'],
            'Condition': ['C1', 'C1'],
            'Quantity': [10.0, np.nan],
        })
        assert assert_constant_within(df, keys=['Proteins', 'Condition'], cols=['Quantity']) is None

    def test_inconsistent_group_raises(self):
        df = pd.DataFrame({
            'Proteins': ['P1', 'P1', 'P2', 'P2'],
            'Condition': ['C1', 'C1', 'C1', 'C1'],
            'Quantity': [10.0, 11.0, 30.0, 30.0],
        })
        with pytest.raises(ValueError) as excinfo:
            assert_constant_within(df, keys=['Proteins', 'Condition'], cols=['Quantity'])
        assert "'Proteins', 'Condition'" in str(excinfo.value)
        assert '1 group' in str(excinfo.value)

    def test_message_counts_all_violating_groups(self):
        df = pd.DataFrame({
            'Proteins': ['P1', 'P1', 'P2', 'P2'],
            'Condition': ['C1', 'C1', 'C1', 'C1'],
            'Quantity': [10.0, 11.0, 30.0, 31.0],
        })
        with pytest.raises(ValueError) as excinfo:
            assert_constant_within(df, keys=['Proteins', 'Condition'], cols=['Quantity'])
        assert '2 group' in str(excinfo.value)


class TestLoadSpectronautProteins:
    def write_report(self, tmp_path, df, name='report.tsv'):
        path = tmp_path / name
        df.to_csv(path, sep='\t', index=False)
        return path

    def test_long_report_with_repeated_precursor_rows(self, tmp_path):
        report = pd.DataFrame({
            'R.Condition': ['C1', 'C1', 'C2', 'C2', 'C1', 'C2'],
            'PG.ProteinGroups': ['P1', 'P1', 'P1', 'P1', 'P2', 'P2'],
            'PG.Genes': ['G1', 'G1', 'G1', 'G1', 'G2', 'G2'],
            'EG.PrecursorId': ['A', 'B', 'A', 'B', 'C', 'C'],
            'PG.Quantity': [10.0, 10.0, 20.0, 20.0, 30.0, 40.0],
        })
        path = self.write_report(tmp_path, report)
        result = load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'], max_cols=['Raw C1', 'Raw C2'])
        expected_result = pd.DataFrame({
            'Proteins': ['P1', 'P2'],
            'N duplicates': [1, 1],
            'Genes': ['G1', 'G2'],
            'Raw C1': [10.0, 30.0],
            'Raw C2': [20.0, 40.0],
        })
        assert result.equals(expected_result)

    def test_pivot_report_takes_the_wide_path(self, tmp_path):
        report = pd.DataFrame({
            'PG.ProteinGroups': ['P1', 'P2'],
            'PG.Genes': ['G1', 'G2'],
            '[1] C1.PG.Quantity': [10.0, 30.0],
            '[2] C2.PG.Quantity': [20.0, 40.0],
        })
        path = self.write_report(tmp_path, report)
        result = load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'], max_cols=['Raw C1', 'Raw C2'])
        expected_result = pd.DataFrame({
            'Proteins': ['P1', 'P2'],
            'N duplicates': [1, 1],
            'Genes': ['G1', 'G2'],
            'Raw C1': [10.0, 30.0],
            'Raw C2': [20.0, 40.0],
        })
        assert result.equals(expected_result)

    def test_missing_genes_falls_back_to_proteins(self, tmp_path):
        report = pd.DataFrame({
            'R.Condition': ['C1', 'C2'],
            'PG.ProteinGroups': ['P1', 'P1'],
            'PG.Quantity': [10.0, 20.0],
        })
        path = self.write_report(tmp_path, report)
        result = load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'], max_cols=['Raw C1', 'Raw C2'])
        expected_result = pd.DataFrame({
            'Proteins': ['P1'],
            'N duplicates': [1],
            'Genes': ['P1'],
            'Raw C1': [10.0],
            'Raw C2': [20.0],
        })
        assert result.equals(expected_result)

    def test_decoy_rows_are_dropped(self, tmp_path):
        report = pd.DataFrame({
            'R.Condition': ['C1', 'C2', 'C1', 'C2'],
            'PG.ProteinGroups': ['P1', 'P1', 'DECOY', 'DECOY'],
            'PG.Genes': ['G1', 'G1', 'DECOY', 'DECOY'],
            'EG.IsDecoy': ['False', 'False', 'True', 'True'],
            'PG.Quantity': [10.0, 20.0, 99.0, 99.0],
        })
        path = self.write_report(tmp_path, report)
        result = load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'], max_cols=['Raw C1', 'Raw C2'])
        expected_result = pd.DataFrame({
            'Proteins': ['P1'],
            'N duplicates': [1],
            'Genes': ['G1'],
            'Raw C1': [10.0],
            'Raw C2': [20.0],
        })
        assert result.equals(expected_result)

    def test_decoy_row_with_divergent_quantity_does_not_trip_the_assertion(self, tmp_path):
        # This test pins the ordering in the design: clean_rows must run before the constancy
        # assertion, or an elution-group decoy row makes a perfectly consistent protein group fail.
        report = pd.DataFrame({
            'R.Condition': ['C1', 'C1', 'C2'],
            'PG.ProteinGroups': ['P1', 'P1', 'P1'],
            'PG.Genes': ['G1', 'G1', 'G1'],
            'EG.IsDecoy': ['False', 'True', 'False'],
            'PG.Quantity': [10.0, 99.0, 20.0],
        })
        path = self.write_report(tmp_path, report)
        result = load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'], max_cols=['Raw C1', 'Raw C2'])
        expected_result = pd.DataFrame({
            'Proteins': ['P1'],
            'N duplicates': [1],
            'Genes': ['G1'],
            'Raw C1': [10.0],
            'Raw C2': [20.0],
        })
        assert result.equals(expected_result)

    def test_report_without_decoy_column_parses_unchanged(self, tmp_path):
        report = pd.DataFrame({
            'R.Condition': ['C1', 'C2'],
            'PG.ProteinGroups': ['P1', 'P1'],
            'PG.Genes': ['G1', 'G1'],
            'PG.Quantity': [10.0, 20.0],
        })
        path = self.write_report(tmp_path, report)
        result = load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'], max_cols=['Raw C1', 'Raw C2'])
        expected_result = pd.DataFrame({
            'Proteins': ['P1'],
            'N duplicates': [1],
            'Genes': ['G1'],
            'Raw C1': [10.0],
            'Raw C2': [20.0],
        })
        assert result.equals(expected_result)

    def test_inconsistent_quantity_raises(self, tmp_path):
        report = pd.DataFrame({
            'R.Condition': ['C1', 'C1', 'C2'],
            'PG.ProteinGroups': ['P1', 'P1', 'P1'],
            'PG.Genes': ['G1', 'G1', 'G1'],
            'PG.Quantity': [10.0, 11.0, 20.0],
        })
        path = self.write_report(tmp_path, report)
        with pytest.raises(ValueError) as excinfo:
            load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'], max_cols=['Raw C1', 'Raw C2'])
        assert '1 group' in str(excinfo.value)

    def test_neither_long_nor_pivot_raises(self, tmp_path):
        report = pd.DataFrame({'PG.ProteinGroups': ['P1'], 'PG.Genes': ['G1']})
        path = self.write_report(tmp_path, report)
        with pytest.raises(ValueError) as excinfo:
            load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'], max_cols=['Raw C1'])
        assert 'R.Condition' in str(excinfo.value)
        assert 'PG.ProteinGroups' in str(excinfo.value)

    def test_missing_required_column_raises(self, tmp_path):
        report = pd.DataFrame({'R.Condition': ['C1'], 'PG.Genes': ['G1'], 'PG.Quantity': [10.0]})
        path = self.write_report(tmp_path, report)
        with pytest.raises(ValueError) as excinfo:
            load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'], max_cols=['Raw C1'])
        assert 'Proteins' in str(excinfo.value)
        assert 'PG.ProteinGroups' in str(excinfo.value)

    def test_empty_condition_raises(self, tmp_path):
        report = pd.DataFrame({
            'R.Condition': ['C1', None],
            'PG.ProteinGroups': ['P1', 'P1'],
            'PG.Genes': ['G1', 'G1'],
            'PG.Quantity': [10.0, 10.0],
        })
        path = self.write_report(tmp_path, report)
        with pytest.raises(ValueError) as excinfo:
            load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'], max_cols=['Raw C1'])
        assert 'condition setup' in str(excinfo.value)

    def test_peptide_quantity_column_is_ignored(self, tmp_path):
        # A precursor-level report carries both roll-ups. Both share the canonical name 'Quantity',
        # so the other level's column must be dropped before the frame is renamed.
        report = pd.DataFrame({
            'R.Condition': ['C1', 'C1', 'C2', 'C2'],
            'PG.ProteinGroups': ['P1', 'P1', 'P1', 'P1'],
            'PG.Genes': ['G1', 'G1', 'G1', 'G1'],
            'PEP.GroupingKey': ['AAAK', 'BBBK', 'AAAK', 'BBBK'],
            'PEP.Quantity': [1.0, 2.0, 3.0, 4.0],
            'PG.Quantity': [10.0, 10.0, 20.0, 20.0],
        })
        path = self.write_report(tmp_path, report)
        result = load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'], max_cols=['Raw C1', 'Raw C2'])
        expected_result = pd.DataFrame({
            'Proteins': ['P1'],
            'N duplicates': [1],
            'Genes': ['G1'],
            'Raw C1': [10.0],
            'Raw C2': [20.0],
        })
        assert result.equals(expected_result)


class TestLoadSpectronautPeptides:
    def write_report(self, tmp_path, df, name='report.tsv'):
        path = tmp_path / name
        df.to_csv(path, sep='\t', index=False)
        return path

    def test_repeated_charges_collapse_to_one_row(self, tmp_path):
        report = pd.DataFrame({
            'R.Condition': ['C1', 'C1', 'C2', 'C2'],
            'PG.ProteinGroups': ['P1', 'P1', 'P1', 'P1'],
            'PG.Genes': ['G1', 'G1', 'G1', 'G1'],
            'PEP.GroupingKey': ['_C[Carbamidomethyl (C)]PEPTIDER_'] * 4,
            'PEP.GroupingKeyType': ['ModifiedSequence'] * 4,
            'FG.Charge': [2, 3, 2, 3],
            'PEP.Quantity': [5.0, 5.0, 7.0, 7.0],
        })
        path = self.write_report(tmp_path, report)
        result = load_spectronaut_dia_peptides(path, '19.9', unique_cols=['Modified sequence'], first_cols=['Proteins', 'Genes'], max_cols=['Raw C1', 'Raw C2'])
        expected_result = pd.DataFrame({
            'Modified sequence': ['CPEPTIDER'],
            'N duplicates': [1],
            'Proteins': ['P1'],
            'Genes': ['G1'],
            'Raw C1': [5.0],
            'Raw C2': [7.0],
        })
        assert result.equals(expected_result)

    def test_stripped_grouping_key_parses(self, tmp_path):
        report = pd.DataFrame({
            'R.Condition': ['C1', 'C2'],
            'PG.ProteinGroups': ['P1', 'P1'],
            'PG.Genes': ['G1', 'G1'],
            'PEP.GroupingKey': ['CPEPTIDER', 'CPEPTIDER'],
            'PEP.GroupingKeyType': ['StrippedSequence', 'StrippedSequence'],
            'PEP.Quantity': [5.0, 7.0],
        })
        path = self.write_report(tmp_path, report)
        result = load_spectronaut_dia_peptides(path, '19.9', unique_cols=['Modified sequence'], first_cols=['Proteins', 'Genes'], max_cols=['Raw C1', 'Raw C2'])
        expected_result = pd.DataFrame({
            'Modified sequence': ['CPEPTIDER'],
            'N duplicates': [1],
            'Proteins': ['P1'],
            'Genes': ['G1'],
            'Raw C1': [5.0],
            'Raw C2': [7.0],
        })
        assert result.equals(expected_result)

    def test_protein_quantity_alone_raises(self, tmp_path):
        # PG.Quantity is a protein roll-up. Reading it as the peptide quantity would repeat one
        # protein measurement under every peptide of that protein, so it must fail loudly.
        report = pd.DataFrame({
            'R.Condition': ['C1', 'C2'],
            'PG.ProteinGroups': ['P1', 'P1'],
            'PG.Genes': ['G1', 'G1'],
            'PEP.GroupingKey': ['CPEPTIDER', 'CPEPTIDER'],
            'PG.Quantity': [5.0, 7.0],
        })
        path = self.write_report(tmp_path, report)
        with pytest.raises(ValueError) as excinfo:
            load_spectronaut_dia_peptides(path, '19.9', unique_cols=['Modified sequence'], first_cols=['Proteins', 'Genes'], max_cols=['Raw C1', 'Raw C2'])
        assert 'Quantity' in str(excinfo.value)
        assert 'PEP.Quantity' in str(excinfo.value)


class TestLoadSpectronautParquet:
    def test_parquet_long_report(self, tmp_path):
        pytest.importorskip('pyarrow')
        report = pd.DataFrame({
            'R_Condition': ['C1', 'C1', 'C1', 'C2'],
            'PG_ProteinGroups': ['P1', 'P1', 'P2', 'P2'],
            'PG_Genes': ['G1', 'G1', 'G2', 'G2'],
            'EG_IsDecoy': [False, True, False, False],
            'PG_Quantity': [10.0, 99.0, 30.0, 40.0],
        })
        path = tmp_path / 'report.parquet'
        report.to_parquet(path, index=False)
        result = load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'], max_cols=['Raw C1', 'Raw C2'])
        expected_result = pd.DataFrame({
            'Proteins': ['P1', 'P2'],
            'N duplicates': [1, 1],
            'Genes': ['G1', 'G2'],
            'Raw C1': [10.0, 30.0],
            'Raw C2': [np.nan, 40.0],
        })
        assert result.equals(expected_result)
