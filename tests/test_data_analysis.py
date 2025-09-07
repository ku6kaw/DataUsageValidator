import unittest
import pandas as pd
import os
from unittest.mock import patch, MagicMock

from src.data_analysis import (
    load_and_preprocess_data_papers,
    filter_by_citation_count,
    analyze_citation_statistics,
    categorize_citations
)
from src.config import OUTPUT_FILE_DATA_PAPERS

class TestDataAnalysis(unittest.TestCase):

    def setUp(self):
        # テスト用のダミーCSVファイルを作成
        self.test_csv_path = "test_data_papers.csv"
        self.dummy_data = {
            'eid': ['eid1', 'eid2', 'eid3', 'eid4', 'eid5'],
            'doi': ['doi1', 'doi2', 'doi3', 'doi4', 'doi5'],
            'title': ['title1', 'title2', 'title3', 'title4', 'title5'],
            'publication_year': ['2020', '2021', '2022', '2023', '2024'],
            'citedby_count': [1, 5, 15, 60, 200]
        }
        self.df_dummy = pd.DataFrame(self.dummy_data)
        self.df_dummy.to_csv(self.test_csv_path, index=False)

        self.empty_csv_path = "empty_data_papers.csv"
        pd.DataFrame().to_csv(self.empty_csv_path, index=False)

    def tearDown(self):
        # テスト用ファイルを削除
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)
        if os.path.exists(self.empty_csv_path):
            os.remove(self.empty_csv_path)

    def test_load_and_preprocess_data_papers_success(self):
        df = load_and_preprocess_data_papers(self.test_csv_path)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 5)
        self.assertTrue(pd.api.types.is_numeric_dtype(df['citedby_count']))

    def test_load_and_preprocess_data_papers_file_not_found(self):
        df = load_and_preprocess_data_papers("non_existent_file.csv")
        self.assertTrue(df.empty)

    def test_filter_by_citation_count(self):
        df = load_and_preprocess_data_papers(self.test_csv_path)
        df_filtered = filter_by_citation_count(df, min_citations=10)
        self.assertEqual(len(df_filtered), 3) # 15, 60, 200

    @patch('builtins.print')
    def test_analyze_citation_statistics(self, mock_print):
        df = load_and_preprocess_data_papers(self.test_csv_path)
        analyze_citation_statistics(df)
        mock_print.assert_called() # printが呼ばれたことを確認

    def test_categorize_citations(self):
        df = load_and_preprocess_data_papers(self.test_csv_path)
        df_filtered = filter_by_citation_count(df, min_citations=2) # 5, 15, 60, 200
        df_categorized = categorize_citations(df_filtered)
        self.assertIn('citation_category', df_categorized.columns)
        self.assertEqual(df_categorized.loc[df_categorized['citedby_count'] == 5, 'citation_category'].iloc[0], '2-10 (Low)')
        self.assertEqual(df_categorized.loc[df_categorized['citedby_count'] == 15, 'citation_category'].iloc[0], '11-50 (Medium)')
        self.assertEqual(df_categorized.loc[df_categorized['citedby_count'] == 60, 'citation_category'].iloc[0], '51-150 (High)')
        self.assertEqual(df_categorized.loc[df_categorized['citedby_count'] == 200, 'citation_category'].iloc[0], '151+ (Top Tier)')

    def test_categorize_citations_empty_df(self):
        df = pd.DataFrame()
        df_categorized = categorize_citations(df)
        self.assertTrue(df_categorized.empty)

if __name__ == '__main__':
    unittest.main()
