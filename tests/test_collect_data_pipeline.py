import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
from pipeline.collect_data_pipeline import run_collect_data_pipeline
from src.config import SCOPUS_API_KEY, SCOPUS_QUERY_DATA_PAPERS, OUTPUT_FILE_DATA_PAPERS, OUTPUT_DIR_PROCESSED

class TestCollectDataPipeline(unittest.TestCase):

    @patch('pipeline.collect_data_pipeline.get_total_data_papers_count')
    @patch('pipeline.collect_data_pipeline.collect_data_papers')
    @patch('pipeline.collect_data_pipeline.save_data_papers_to_csv')
    def test_run_collect_data_pipeline_success(self, mock_save, mock_collect, mock_get_total):
        """
        データ論文収集パイプラインが正常に実行される場合のテスト。
        """
        mock_get_total.return_value = 100
        mock_collect.return_value = pd.DataFrame({
            'eid': [f'eid{i}' for i in range(100)],
            'doi': [f'doi{i}' for i in range(100)],
            'title': [f'title{i}' for i in range(100)],
            'publication_year': ['2020'] * 100,
            'citedby_count': ['10'] * 100
        })

        run_collect_data_pipeline(
            api_key="TEST_KEY",
            query="TEST_QUERY",
            output_file="test_data_papers.csv",
            output_dir="test_processed"
        )

        mock_get_total.assert_called_once_with(api_key="TEST_KEY", query="TEST_QUERY")
        mock_collect.assert_called_once_with(api_key="TEST_KEY", query="TEST_QUERY", total_results=100)
        mock_save.assert_called_once()
        self.assertFalse(mock_save.call_args[0][0].empty) # DataFrameが空でないことを確認

    @patch('pipeline.collect_data_pipeline.get_total_data_papers_count')
    @patch('pipeline.collect_data_pipeline.collect_data_papers')
    @patch('pipeline.collect_data_pipeline.save_data_papers_to_csv')
    def test_run_collect_data_pipeline_no_results(self, mock_save, mock_collect, mock_get_total):
        """
        総件数が0の場合、パイプラインが適切にスキップされるテスト。
        """
        mock_get_total.return_value = 0

        run_collect_data_pipeline(
            api_key="TEST_KEY",
            query="TEST_QUERY",
            output_file="test_data_papers.csv",
            output_dir="test_processed"
        )

        mock_get_total.assert_called_once_with(api_key="TEST_KEY", query="TEST_QUERY")
        mock_collect.assert_not_called()
        mock_save.assert_not_called()

    @patch('pipeline.collect_data_pipeline.get_total_data_papers_count')
    @patch('pipeline.collect_data_pipeline.collect_data_papers')
    @patch('pipeline.collect_data_pipeline.save_data_papers_to_csv')
    def test_run_collect_data_pipeline_empty_df_after_collection(self, mock_save, mock_collect, mock_get_total):
        """
        総件数はあるが、収集結果が空のDataFrameの場合のテスト。
        """
        mock_get_total.return_value = 50
        mock_collect.return_value = pd.DataFrame() # 空のDataFrameを返す

        run_collect_data_pipeline(
            api_key="TEST_KEY",
            query="TEST_QUERY",
            output_file="test_data_papers.csv",
            output_dir="test_processed"
        )

        mock_get_total.assert_called_once_with(api_key="TEST_KEY", query="TEST_QUERY")
        mock_collect.assert_called_once_with(api_key="TEST_KEY", query="TEST_QUERY", total_results=50)
        mock_save.assert_not_called() # 空のDataFrameは保存されない

if __name__ == '__main__':
    unittest.main()
