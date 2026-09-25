import pandas as pd
import numpy as np
import pytest

from curve_curator.search_engine_outputs.Spectronaut import SpectronautMap, _to_parquet_name
from curve_curator.data_parser import assert_constant_within, load, load_spectronaut_dia_proteins, load_spectronaut_dia_peptides


class TestSpectronautMap:
    def test_rename_general_columns_csv_dialect(self):
        cols = pd.Index(['R.Label', 'PG.ProteinGroups', 'PG.Genes', 'PEP.GroupingKey', 'PEP.GroupingKeyType', 'EG.IsDecoy', 'PG.Quantity'])
        expected_result = pd.Index(['Run', 'Proteins', 'Genes', 'Modified sequence', 'Grouping type', 'Decoy', 'Quantity'])
        assert SpectronautMap.rename_general_columns(cols).equals(expected_result)

    def test_rename_general_columns_parquet_dialect(self):
        cols = pd.Index(['R_Label', 'PG_ProteinGroups', 'PG_Genes', 'PEP_GroupingKey', 'PEP_GroupingKeyType', 'EG_IsDecoy', 'PEP_Quantity'])
        expected_result = pd.Index(['Run', 'Proteins', 'Genes', 'Modified sequence', 'Grouping type', 'Decoy', 'Quantity'])
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


class TestToParquetName:
    def test_single_dot(self):
        assert _to_parquet_name('PG.Quantity') == 'PG_Quantity'
        assert _to_parquet_name('R.Condition') == 'R_Condition'

    def test_every_dot_and_space_is_replaced(self):
        assert _to_parquet_name('[1] CondA.PG.Quantity') == '[1]_CondA_PG_Quantity'


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
        cols = pd.Index(['PG.Quantity', 'PEP.GroupingKey', 'R.Label'])
        expected_result = pd.Index(['PG.Quantity', 'PEP.GroupingKey', 'R.Label'])
        assert SpectronautMap.rename_quantity_columns(cols).equals(expected_result)

    def test_parquet_dialect_pivot_header(self):
        cols = pd.Index(['PG_Genes', 'PG_ProteinGroups', '[1]_DMSO_vs_DMSO_plate1_rep3_PG_Quantity'])
        expected_result = pd.Index(['PG_Genes', 'PG_ProteinGroups', 'Raw DMSO_vs_DMSO_plate1_rep3'])
        assert SpectronautMap.rename_quantity_columns(cols).equals(expected_result)

    def test_parquet_dialect_peptide_pivot_header(self):
        cols = pd.Index(['PEP_GroupingKey', '[10]_Cond_A_PEP_Quantity'])
        expected_result = pd.Index(['PEP_GroupingKey', 'Raw Cond_A'])
        assert SpectronautMap.rename_quantity_columns(cols).equals(expected_result)

    def test_parquet_long_quantity_columns_are_not_pivot_headers(self):
        cols = pd.Index(['PG_Quantity', 'PEP_Quantity', 'R_Label'])
        expected_result = pd.Index(['PG_Quantity', 'PEP_Quantity', 'R_Label'])
        assert SpectronautMap.rename_quantity_columns(cols).equals(expected_result)

    def test_parquet_runs_restore_spaces_dots_and_underscores_from_experiments(self):
        # Keep literal headers here to test the production normalizer independently.
        cols = pd.Index(['[1]_Cond_A_1_PG_Quantity', '[2]_Cond_A_2_PG_Quantity'])
        experiments = ['Cond A.1', 'Cond_A.2']
        expected_result = pd.Index(['Raw Cond A.1', 'Raw Cond_A.2'])
        assert SpectronautMap.rename_quantity_columns(
            cols, experiments=experiments, parquet=True).equals(expected_result)

    def test_tsv_run_matching_stays_exact(self):
        cols = pd.Index(['[1] Cond_A_1.PG.Quantity'])
        result = SpectronautMap.rename_quantity_columns(
            cols, experiments=['Cond A.1'], parquet=False)
        assert result.equals(pd.Index(['Raw Cond_A_1']))


class TestRestructureLongReport:
    def test_long_to_wide(self):
        df = pd.DataFrame({
            'Proteins': ['P1', 'P1', 'P2', 'P2'],
            'Genes': ['G1', 'G1', 'G2', 'G2'],
            'Run': ['C1', 'C2', 'C1', 'C2'],
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

    def test_missing_run_becomes_nan(self):
        df = pd.DataFrame({
            'Proteins': ['P1', 'P1', 'P2'],
            'Genes': ['G1', 'G1', 'G2'],
            'Run': ['C1', 'C2', 'C1'],
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
            'Run': ['C1', 'C2', 'C1', 'C2'],
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
            'Run': ['C1', 'C1', 'C1', 'C2'],
            'Quantity': [10.0, 10.0, 30.0, 40.0],
        })
        assert assert_constant_within(df, keys=['Proteins', 'Run'], cols=['Quantity']) is None

    def test_missing_values_do_not_count(self):
        df = pd.DataFrame({
            'Proteins': ['P1', 'P1'],
            'Run': ['C1', 'C1'],
            'Quantity': [10.0, np.nan],
        })
        assert assert_constant_within(df, keys=['Proteins', 'Run'], cols=['Quantity']) is None

    def test_inconsistent_group_raises(self):
        df = pd.DataFrame({
            'Proteins': ['P1', 'P1', 'P2', 'P2'],
            'Run': ['C1', 'C1', 'C1', 'C1'],
            'Quantity': [10.0, 11.0, 30.0, 30.0],
        })
        with pytest.raises(ValueError) as excinfo:
            assert_constant_within(df, keys=['Proteins', 'Run'], cols=['Quantity'])
        assert "'Proteins', 'Run'" in str(excinfo.value)
        assert '1 group' in str(excinfo.value)

    def test_message_counts_all_violating_groups(self):
        df = pd.DataFrame({
            'Proteins': ['P1', 'P1', 'P2', 'P2'],
            'Run': ['C1', 'C1', 'C1', 'C1'],
            'Quantity': [10.0, 11.0, 30.0, 31.0],
        })
        with pytest.raises(ValueError) as excinfo:
            assert_constant_within(df, keys=['Proteins', 'Run'], cols=['Quantity'])
        assert '2 group' in str(excinfo.value)


class TestLoadSpectronautProteins:
    def write_report(self, tmp_path, df, name='report.tsv'):
        path = tmp_path / name
        df.to_csv(path, sep='\t', index=False)
        return path

    def test_long_report_with_repeated_precursor_rows(self, tmp_path):
        report = pd.DataFrame({
            'R.Label': ['C1', 'C1', 'C2', 'C2', 'C1', 'C2'],
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
            'R.Label': ['C1', 'C2'],
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
            'R.Label': ['C1', 'C2', 'C1', 'C2'],
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
        report = pd.DataFrame({
            'R.Label': ['C1', 'C1', 'C2'],
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
            'R.Label': ['C1', 'C2'],
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
            'R.Label': ['C1', 'C1', 'C2'],
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
        assert 'R.Label' in str(excinfo.value)
        assert 'PG.ProteinGroups' in str(excinfo.value)

    def test_missing_required_column_raises(self, tmp_path):
        report = pd.DataFrame({'R.Label': ['C1'], 'PG.Genes': ['G1'], 'PG.Quantity': [10.0]})
        path = self.write_report(tmp_path, report)
        with pytest.raises(ValueError) as excinfo:
            load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'], max_cols=['Raw C1'])
        assert 'Proteins' in str(excinfo.value)
        assert 'PG.ProteinGroups' in str(excinfo.value)

    def test_empty_run_label_raises(self, tmp_path):
        report = pd.DataFrame({
            'R.Label': ['C1', None],
            'PG.ProteinGroups': ['P1', 'P1'],
            'PG.Genes': ['G1', 'G1'],
            'PG.Quantity': [10.0, 10.0],
        })
        path = self.write_report(tmp_path, report)
        with pytest.raises(ValueError) as excinfo:
            load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'], max_cols=['Raw C1'])
        assert 'condition setup' in str(excinfo.value)

    def test_peptide_quantity_column_is_ignored(self, tmp_path):
        report = pd.DataFrame({
            'R.Label': ['C1', 'C1', 'C2', 'C2'],
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
            'R.Label': ['C1', 'C1', 'C2', 'C2'],
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
            'R.Label': ['C1', 'C2'],
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
        report = pd.DataFrame({
            'R.Label': ['C1', 'C2'],
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
            'R_Label': ['C1', 'C1', 'C1', 'C2'],
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


#
# Headers and rows are trimmed from Spectronaut 19.9.250324 exports.

SPECTRONAUT_LONG_COLUMNS = ['R.Label', 'PG.Genes', 'PG.ProteinGroups', 'PG.Quantity',
                            'PEP.GroupingKey', 'PEP.GroupingKeyType', 'PEP.Quantity', 'EG.IsDecoy']

SPECTRONAUT_LONG_ROWS = [
    ['dose_0', 'USP36', 'A0A075B784;Q9P275', 446.7162780761719, 'ALELFVK', 'Stripped Sequence', 672.8201293945312, 'False'],
    ['dose_0', 'USP36', 'A0A075B784;Q9P275', 446.7162780761719, 'EGQAQLPAVR', 'Stripped Sequence', 539.1575927734375, 'False'],
    ['dose_1', 'USP36', 'A0A075B784;Q9P275', 893.4325561523438, 'ALELFVK', 'Stripped Sequence', 1345.6402587890625, 'False'],
    ['dose_1', 'USP36', 'A0A075B784;Q9P275', 893.4325561523438, 'EGQAQLPAVR', 'Stripped Sequence', 1078.315185546875, 'False'],
    ['dose_0', 'ICOSLG', 'A0A087X1L8;O75144', 429.8607482910156, 'GLYDVVSVLR', 'Stripped Sequence', 429.8607482910156, 'False'],
    ['dose_1', 'ICOSLG', 'A0A087X1L8;O75144', 214.9303741455078, 'GLYDVVSVLR', 'Stripped Sequence', 214.9303741455078, 'False'],
    ['dose_0', 'TRYP', 'P00761', 159657.140625, 'LGEHNIDVLEGNEQFINAAK', 'Stripped Sequence', 109553.109375, 'False'],
    ['dose_0', 'TRYP', 'P00761', 159657.140625, 'LGEHNIDVLEGNEQFINAAK', 'Stripped Sequence', 109553.109375, 'False'],
    ['dose_1', 'TRYP', 'P00761', 79828.5703125, 'LGEHNIDVLEGNEQFINAAK', 'Stripped Sequence', 54776.5546875, 'False'],
    ['dose_0', 'DECOY', 'DECOY_A0A075B784', 99999.0, 'KVFLEAL', 'Stripped Sequence', 99999.0, 'True'],
    ['dose_1', 'DECOY', 'DECOY_A0A075B784', 99999.0, 'KVFLEAL', 'Stripped Sequence', 99999.0, 'True'],
]

EXPECTED_PROTEINS = pd.DataFrame({
    'Proteins': ['A0A075B784;Q9P275', 'A0A087X1L8;O75144', 'P00761'],
    'N duplicates': [1, 1, 1],
    'Genes': ['USP36', 'ICOSLG', 'TRYP'],
    'Raw dose_0': [446.7162780761719, 429.8607482910156, 159657.140625],
    'Raw dose_1': [893.4325561523438, 214.9303741455078, 79828.5703125],
})

EXPECTED_PEPTIDES = pd.DataFrame({
    'Modified sequence': ['ALELFVK', 'EGQAQLPAVR', 'GLYDVVSVLR', 'LGEHNIDVLEGNEQFINAAK'],
    'N duplicates': [1, 1, 1, 1],
    'Proteins': ['A0A075B784;Q9P275', 'A0A075B784;Q9P275', 'A0A087X1L8;O75144', 'P00761'],
    'Genes': ['USP36', 'USP36', 'ICOSLG', 'TRYP'],
    'Raw dose_0': [672.8201293945312, 539.1575927734375, 429.8607482910156, 109553.109375],
    'Raw dose_1': [1345.6402587890625, 1078.315185546875, 214.9303741455078, 54776.5546875],
})


def write_long_tsv(path):
    df = pd.DataFrame(SPECTRONAUT_LONG_ROWS, columns=SPECTRONAUT_LONG_COLUMNS)
    df.to_csv(path, sep='\t', index=False)
    return path


def write_long_parquet(path):
    df = pd.DataFrame(SPECTRONAUT_LONG_ROWS, columns=SPECTRONAUT_LONG_COLUMNS)
    df.columns = [_to_parquet_name(c) for c in df.columns]
    df['EG_IsDecoy'] = df['EG_IsDecoy'] == 'True'
    df.to_parquet(path, index=False)
    return path


class TestLongReportEndToEnd:
    def test_protein_level_tsv(self, tmp_path):
        path = write_long_tsv(tmp_path / 'report.tsv')
        result = load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'],
                                               max_cols=['Raw dose_0', 'Raw dose_1'])
        assert result.equals(EXPECTED_PROTEINS)

    def test_protein_level_parquet(self, tmp_path):
        pytest.importorskip('pyarrow')
        path = write_long_parquet(tmp_path / 'report.parquet')
        result = load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'],
                                               max_cols=['Raw dose_0', 'Raw dose_1'])
        assert result.equals(EXPECTED_PROTEINS)

    def test_peptide_level_tsv(self, tmp_path):
        path = write_long_tsv(tmp_path / 'report.tsv')
        result = load_spectronaut_dia_peptides(path, '19.9', unique_cols=['Modified sequence'],
                                               first_cols=['Proteins', 'Genes'],
                                               max_cols=['Raw dose_0', 'Raw dose_1'])
        assert result.equals(EXPECTED_PEPTIDES)

    def test_peptide_level_parquet(self, tmp_path):
        pytest.importorskip('pyarrow')
        path = write_long_parquet(tmp_path / 'report.parquet')
        result = load_spectronaut_dia_peptides(path, '19.9', unique_cols=['Modified sequence'],
                                               first_cols=['Proteins', 'Genes'],
                                               max_cols=['Raw dose_0', 'Raw dose_1'])
        assert result.equals(EXPECTED_PEPTIDES)

    def test_the_two_dialects_agree(self, tmp_path):
        pytest.importorskip('pyarrow')
        tsv = load_spectronaut_dia_proteins(write_long_tsv(tmp_path / 'report.tsv'), '19.9',
                                            unique_cols=['Proteins'], first_cols=['Genes'],
                                            max_cols=['Raw dose_0', 'Raw dose_1'])
        parquet = load_spectronaut_dia_proteins(write_long_parquet(tmp_path / 'report.parquet'), '19.9',
                                                unique_cols=['Proteins'], first_cols=['Genes'],
                                                max_cols=['Raw dose_0', 'Raw dose_1'])
        assert tsv.equals(parquet)


SPECTRONAUT_PROTEIN_PIVOT_COLUMNS = ['PG.Genes', 'PG.ProteinGroups', '[1] dose_0.PG.Quantity', '[2] dose_1.PG.Quantity']
SPECTRONAUT_PROTEIN_PIVOT_ROWS = [
    ['USP36', 'A0A075B784;Q9P275', 446.7162780761719, 893.4325561523438],
    ['ICOSLG', 'A0A087X1L8;O75144', 429.8607482910156, 214.9303741455078],
    ['TRYP', 'P00761', 159657.140625, 79828.5703125],
]

SPECTRONAUT_PEPTIDE_PIVOT_COLUMNS = ['PG.Genes', 'PG.ProteinGroups', 'PEP.GroupingKey',
                                     '[1] dose_0.PEP.Quantity', '[2] dose_1.PEP.Quantity']
SPECTRONAUT_PEPTIDE_PIVOT_ROWS = [
    ['USP36', 'A0A075B784;Q9P275', 'ALELFVK', 672.8201293945312, 1345.6402587890625],
    ['USP36', 'A0A075B784;Q9P275', 'EGQAQLPAVR', 539.1575927734375, 1078.315185546875],
    ['ICOSLG', 'A0A087X1L8;O75144', 'GLYDVVSVLR', 429.8607482910156, 214.9303741455078],
    ['TRYP', 'P00761', 'LGEHNIDVLEGNEQFINAAK', 109553.109375, 54776.5546875],
]


def write_pivot(path, columns, rows, parquet=False):
    df = pd.DataFrame(rows, columns=columns)
    if parquet:
        df.columns = [_to_parquet_name(c) for c in df.columns]
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, sep='\t', index=False)
    return path


class TestPivotReportEndToEnd:
    def test_protein_level_tsv(self, tmp_path):
        path = write_pivot(tmp_path / 'pivot.tsv', SPECTRONAUT_PROTEIN_PIVOT_COLUMNS, SPECTRONAUT_PROTEIN_PIVOT_ROWS)
        result = load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'],
                                               max_cols=['Raw dose_0', 'Raw dose_1'])
        assert result.equals(EXPECTED_PROTEINS)

    def test_protein_level_parquet(self, tmp_path):
        pytest.importorskip('pyarrow')
        path = write_pivot(tmp_path / 'pivot.parquet', SPECTRONAUT_PROTEIN_PIVOT_COLUMNS,
                           SPECTRONAUT_PROTEIN_PIVOT_ROWS, parquet=True)
        result = load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'],
                                               max_cols=['Raw dose_0', 'Raw dose_1'])
        assert result.equals(EXPECTED_PROTEINS)

    def test_peptide_level_tsv(self, tmp_path):
        path = write_pivot(tmp_path / 'pivot.tsv', SPECTRONAUT_PEPTIDE_PIVOT_COLUMNS, SPECTRONAUT_PEPTIDE_PIVOT_ROWS)
        result = load_spectronaut_dia_peptides(path, '19.9', unique_cols=['Modified sequence'],
                                               first_cols=['Proteins', 'Genes'],
                                               max_cols=['Raw dose_0', 'Raw dose_1'])
        assert result.equals(EXPECTED_PEPTIDES)

    def test_peptide_level_parquet(self, tmp_path):
        pytest.importorskip('pyarrow')
        path = write_pivot(tmp_path / 'pivot.parquet', SPECTRONAUT_PEPTIDE_PIVOT_COLUMNS,
                           SPECTRONAUT_PEPTIDE_PIVOT_ROWS, parquet=True)
        result = load_spectronaut_dia_peptides(path, '19.9', unique_cols=['Modified sequence'],
                                               first_cols=['Proteins', 'Genes'],
                                               max_cols=['Raw dose_0', 'Raw dose_1'])
        assert result.equals(EXPECTED_PEPTIDES)

    def test_pivot_and_long_reports_of_one_experiment_agree(self, tmp_path):
        long_report = load_spectronaut_dia_proteins(write_long_tsv(tmp_path / 'long.tsv'), '19.9',
                                                    unique_cols=['Proteins'], first_cols=['Genes'],
                                                    max_cols=['Raw dose_0', 'Raw dose_1'])
        pivot = load_spectronaut_dia_proteins(
            write_pivot(tmp_path / 'pivot.tsv', SPECTRONAUT_PROTEIN_PIVOT_COLUMNS, SPECTRONAUT_PROTEIN_PIVOT_ROWS),
            '19.9', unique_cols=['Proteins'], first_cols=['Genes'], max_cols=['Raw dose_0', 'Raw dose_1'])
        assert long_report.equals(pivot)

    def test_pivot_header_without_a_condition_setup_uses_the_run_name(self, tmp_path):
        columns = ['PG.Genes', 'PG.ProteinGroups', '[1] DMSO_vs_DMSO_plate1_rep3.PG.Quantity']
        path = write_pivot(tmp_path / 'pivot.tsv', columns, [['USP36', 'A0A075B784;Q9P275', 446.7162780761719]])
        result = load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'],
                                               max_cols=['Raw DMSO_vs_DMSO_plate1_rep3'])
        expected_result = pd.DataFrame({
            'Proteins': ['A0A075B784;Q9P275'],
            'N duplicates': [1],
            'Genes': ['USP36'],
            'Raw DMSO_vs_DMSO_plate1_rep3': [446.7162780761719],
        })
        assert result.equals(expected_result)


class TestLoadDispatchEndToEnd:
    def config(self, path, data_type):
        return {
            'Paths': {'input_file': path},
            'Experiment': {
                'measurement_type': 'DIA',
                'data_type': data_type,
                'search_engine': 'SPECTRONAUT',
                'search_engine_version': '19.9',
                'experiments': ['dose_0', 'dose_1'],
            },
        }

    def test_protein_level(self, tmp_path):
        path = write_long_tsv(tmp_path / 'report.tsv')
        result = load(self.config(path, 'PROTEIN'))
        expected_result = EXPECTED_PROTEINS.copy()
        expected_result['Name'] = expected_result['Genes']
        assert result.equals(expected_result)

    def test_peptide_level(self, tmp_path):
        path = write_long_tsv(tmp_path / 'report.tsv')
        result = load(self.config(path, 'PEPTIDE'))
        expected_result = EXPECTED_PEPTIDES.copy()
        expected_result['Name'] = expected_result['Genes']
        assert result.equals(expected_result)

    def test_pivot_report_through_the_dispatch(self, tmp_path):
        path = write_pivot(tmp_path / 'pivot.tsv', SPECTRONAUT_PROTEIN_PIVOT_COLUMNS, SPECTRONAUT_PROTEIN_PIVOT_ROWS)
        result = load(self.config(path, 'PROTEIN'))
        expected_result = EXPECTED_PROTEINS.copy()
        expected_result['Name'] = expected_result['Genes']
        assert result.equals(expected_result)

    def test_experiment_missing_from_the_report_exits(self, tmp_path):
        path = write_long_tsv(tmp_path / 'report.tsv')
        config = self.config(path, 'PROTEIN')
        config['Experiment']['experiments'] = ['dose_0', 'dose_1', 'dose_2']
        with pytest.raises(SystemExit):
            load(config)

    def test_extra_runs_in_the_report_are_ignored(self, tmp_path):
        path = write_long_tsv(tmp_path / 'report.tsv')
        config = self.config(path, 'PROTEIN')
        config['Experiment']['experiments'] = ['dose_0']
        result = load(config)
        assert list(result.columns) == ['Proteins', 'N duplicates', 'Genes', 'Raw dose_0', 'Name']
        assert result['Raw dose_0'].tolist() == [446.7162780761719, 429.8607482910156, 159657.140625]


class TestInvalidRunLabels:
    def test_reused_run_label_with_different_quantities_raises(self, tmp_path):
        report = pd.DataFrame([
            ['Not Defined', 'USP36', 'A0A075B784;Q9P275', 446.7162780761719, 'ALELFVK', 'Stripped Sequence', 672.8201293945312, 'False'],
            ['Not Defined', 'USP36', 'A0A075B784;Q9P275', 893.4325561523438, 'ALELFVK', 'Stripped Sequence', 1345.6402587890625, 'False'],
        ], columns=SPECTRONAUT_LONG_COLUMNS)
        path = tmp_path / 'report.tsv'
        report.to_csv(path, sep='\t', index=False)
        with pytest.raises(ValueError) as excinfo:
            load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'],
                                          max_cols=['Raw Not Defined'])
        assert '1 group' in str(excinfo.value)


class TestRunKey:
    def test_condition_column_instead_of_label_raises(self, tmp_path):
        report = pd.DataFrame({
            'R.Condition': ['DMSO_vs_DMSO', 'DMSO_vs_DMSO'],
            'PG.ProteinGroups': ['P1', 'P1'],
            'PG.Genes': ['G1', 'G1'],
            'PG.Quantity': [10.0, 20.0],
        })
        path = tmp_path / 'report.tsv'
        report.to_csv(path, sep='\t', index=False)
        with pytest.raises(ValueError) as excinfo:
            load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'],
                                          max_cols=['Raw DMSO_vs_DMSO'])
        assert 'R.Condition' in str(excinfo.value)
        assert 'R.Label' in str(excinfo.value)
        assert 'per run' in str(excinfo.value)

    def test_several_runs_of_one_condition_stay_separate(self, tmp_path):
        report = pd.DataFrame({
            'R.Label': ['DMSO_rep1', 'DMSO_rep2', 'DMSO_rep3'],
            'PG.Genes': ['USP36', 'USP36', 'USP36'],
            'PG.ProteinGroups': ['A0A075B784;Q9P275'] * 3,
            'PG.Quantity': [446.7162780761719, 512.0, 388.5],
        })
        path = tmp_path / 'report.tsv'
        report.to_csv(path, sep='\t', index=False)
        result = load_spectronaut_dia_proteins(path, '19.9', unique_cols=['Proteins'], first_cols=['Genes'],
                                               max_cols=['Raw DMSO_rep1', 'Raw DMSO_rep2', 'Raw DMSO_rep3'])
        expected_result = pd.DataFrame({
            'Proteins': ['A0A075B784;Q9P275'],
            'N duplicates': [1],
            'Genes': ['USP36'],
            'Raw DMSO_rep1': [446.7162780761719],
            'Raw DMSO_rep2': [512.0],
            'Raw DMSO_rep3': [388.5],
        })
        assert result.equals(expected_result)

    def test_a_long_and_a_pivot_export_name_the_same_experiments(self, tmp_path):
        run = 'DMSO_vs_DMSO_plate1_rep3'
        long_report = pd.DataFrame({
            'R.Label': [run],
            'PG.Genes': ['USP36'],
            'PG.ProteinGroups': ['A0A075B784;Q9P275'],
            'PG.Quantity': [446.7162780761719],
        })
        long_path = tmp_path / 'long.tsv'
        long_report.to_csv(long_path, sep='\t', index=False)
        pivot_path = write_pivot(tmp_path / 'pivot.tsv',
                                 ['PG.Genes', 'PG.ProteinGroups', f'[1] {run}.PG.Quantity'],
                                 [['USP36', 'A0A075B784;Q9P275', 446.7162780761719]])
        kwargs = dict(unique_cols=['Proteins'], first_cols=['Genes'], max_cols=[f'Raw {run}'])
        assert load_spectronaut_dia_proteins(long_path, '19.9', **kwargs).equals(
               load_spectronaut_dia_proteins(pivot_path, '19.9', **kwargs))


class TestParquetRunNamesThroughDispatch:
    run = 'Cond A.1'

    @staticmethod
    def config(path, data_type, experiments=None):
        return {
            'Paths': {'input_file': path},
            'Experiment': {
                'measurement_type': 'DIA',
                'data_type': data_type,
                'search_engine': 'SPECTRONAUT',
                'search_engine_version': '19.9',
                'experiments': experiments or ['Cond A.1'],
            },
        }

    @pytest.mark.parametrize('data_type', ['PROTEIN', 'PEPTIDE'])
    def test_all_four_report_forms_return_the_same_complete_frame(self, tmp_path, data_type):
        pytest.importorskip('pyarrow')

        if data_type == 'PROTEIN':
            long_tsv = pd.DataFrame({
                'R.Label': [self.run],
                'PG.Genes': ['GENE1'],
                'PG.ProteinGroups': ['P1'],
                'PG.Quantity': [123.5],
            })
            pivot_tsv = pd.DataFrame({
                'PG.Genes': ['GENE1'],
                'PG.ProteinGroups': ['P1'],
                '[1] Cond A.1.PG.Quantity': [123.5],
            })
            long_parquet = pd.DataFrame({
                'R_Label': [self.run],
                'PG_Genes': ['GENE1'],
                'PG_ProteinGroups': ['P1'],
                'PG_Quantity': [123.5],
            })
            pivot_parquet = pd.DataFrame({
                'PG_Genes': ['GENE1'],
                'PG_ProteinGroups': ['P1'],
                '[1]_Cond_A_1_PG_Quantity': [123.5],
            })
        else:
            long_tsv = pd.DataFrame({
                'R.Label': [self.run],
                'PG.Genes': ['GENE1'],
                'PG.ProteinGroups': ['P1'],
                'PEP.GroupingKey': ['PEPTIDEK'],
                'PEP.Quantity': [45.25],
            })
            pivot_tsv = pd.DataFrame({
                'PG.Genes': ['GENE1'],
                'PG.ProteinGroups': ['P1'],
                'PEP.GroupingKey': ['PEPTIDEK'],
                '[1] Cond A.1.PEP.Quantity': [45.25],
            })
            long_parquet = pd.DataFrame({
                'R_Label': [self.run],
                'PG_Genes': ['GENE1'],
                'PG_ProteinGroups': ['P1'],
                'PEP_GroupingKey': ['PEPTIDEK'],
                'PEP_Quantity': [45.25],
            })
            pivot_parquet = pd.DataFrame({
                'PG_Genes': ['GENE1'],
                'PG_ProteinGroups': ['P1'],
                'PEP_GroupingKey': ['PEPTIDEK'],
                '[1]_Cond_A_1_PEP_Quantity': [45.25],
            })

        paths = {
            'long TSV': tmp_path / f'{data_type.lower()}-long.tsv',
            'pivot TSV': tmp_path / f'{data_type.lower()}-pivot.tsv',
            'long Parquet': tmp_path / f'{data_type.lower()}-long.parquet',
            'pivot Parquet': tmp_path / f'{data_type.lower()}-pivot.parquet',
        }
        long_tsv.to_csv(paths['long TSV'], sep='\t', index=False)
        pivot_tsv.to_csv(paths['pivot TSV'], sep='\t', index=False)
        long_parquet.to_parquet(paths['long Parquet'], index=False)
        pivot_parquet.to_parquet(paths['pivot Parquet'], index=False)

        results = {name: load(self.config(path, data_type)) for name, path in paths.items()}
        reference = results['long TSV']
        for name, result in results.items():
            assert result.equals(reference), f'{name} did not match the long TSV frame'
            assert f'Raw {self.run}' in result.columns

    def test_configured_names_that_normalize_to_the_same_header_raise(self, tmp_path):
        pytest.importorskip('pyarrow')
        report = pd.DataFrame({
            'PG_Genes': ['GENE1'],
            'PG_ProteinGroups': ['P1'],
            '[1]_Cond_A_PG_Quantity': [123.5],
        })
        path = tmp_path / 'configured-collision.parquet'
        report.to_parquet(path, index=False)

        with pytest.raises(ValueError, match='configured.*ambiguous') as excinfo:
            load(self.config(path, 'PROTEIN', experiments=['Cond A', 'Cond_A']))
        assert 'Cond_A' in str(excinfo.value)

    def test_configured_run_that_collides_with_an_extra_report_run_raises(self, tmp_path):
        pytest.importorskip('pyarrow')
        report = pd.DataFrame({
            'PG_Genes': ['GENE1'],
            'PG_ProteinGroups': ['P1'],
            '[1]_Cond_A_PG_Quantity': [123.5],
            '[2]_Cond_A_PG_Quantity': [456.0],
        })
        path = tmp_path / 'report-collision.parquet'
        report.to_parquet(path, index=False)

        with pytest.raises(ValueError, match='quantity headers.*ambiguous') as excinfo:
            load(self.config(path, 'PROTEIN', experiments=['Cond A']))
        assert '[1]_Cond_A_PG_Quantity' in str(excinfo.value)
        assert '[2]_Cond_A_PG_Quantity' in str(excinfo.value)
