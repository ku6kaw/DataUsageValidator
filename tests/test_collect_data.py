import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.collect_data import main
from src.config import OUTPUT_FILE_DATA_PAPERS

class TestCollectData(unittest.TestCase):

    @patch('src.collect_data.get_total_data_papers_count')
    @patch('src.collect_data.collect_data_papers')
    @patch('src.collect_data.save_data_papers_to_csv')
    def test_main_success(self, mock_save, mock_collect, mock_get_count):
        """
        main関数が成功した場合のテスト。
        """
        mock_get_count.return_value = 100
        mock_collect.return_value = pd.DataFrame([
            {'eid': 'eid1', 'doi': 'doi1', 'title': 'title1', 'publication_year': '2020', 'citedby_count': '10'}
        ])

        main()

        mock_get_count.assert_called_once()
        mock_collect.assert_called_once()
        mock_save.assert_called_once()

    @patch('src.collect_data.get_total_data_papers_count')
    @patch('src.collect_data.collect_data_papers')
    @patch('src.collect_data.save_data_papers_to_csv')
    def test_main_no_total_results(self, mock_save, mock_collect, mock_get_count):
        """
        総件数が0の場合のテスト。
        """
        mock_get_count.return_value = 0

        main()

        mock_get_count.assert_called_once()
        mock_collect.assert_not_called()
        mock_save.assert_not_called()

    @patch('src.collect_data.get_total_data_papers_count')
    @patch('src.collect_data.collect_data_papers')
    @patch('src.collect_data.save_data_papers_to_csv')
    def test_main_empty_dataframe(self, mock_save, mock_collect, mock_get_count):
        """
        データ収集結果が空のDataFrameの場合のテスト。
        """
        mock_get_count.return_value = 100
        mock_collect.return_value = pd.DataFrame()

        main()

        mock_get_count.assert_called_once()
        mock_collect.assert_called_once()
        mock_save.assert_not_called()

if __name__ == '__main__':
    unittest.main()
