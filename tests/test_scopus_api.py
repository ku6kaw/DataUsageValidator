import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import requests # Add this import
from src.scopus_api import get_total_data_papers_count, collect_data_papers, save_data_papers_to_csv
from src.config import SCOPUS_API_KEY, SCOPUS_BASE_URL, SCOPUS_QUERY_DATA_PAPERS, OUTPUT_DIR_PROCESSED, OUTPUT_FILE_DATA_PAPERS

class TestScopusApi(unittest.TestCase):

    @patch('requests.get')
    def test_get_total_data_papers_count_success(self, mock_get):
        """
        総件数取得が成功した場合のテスト。
        """
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'search-results': {
                'opensearch:totalResults': '100'
            }
        }
        mock_get.return_value = mock_response

        count = get_total_data_papers_count(api_key="TEST_KEY", query="TEST_QUERY")
        self.assertEqual(count, 100)
        mock_get.assert_called_once_with(SCOPUS_BASE_URL, params={'apiKey': 'TEST_KEY', 'query': 'TEST_QUERY', 'count': 1})

    @patch('requests.get')
    def test_get_total_data_papers_count_api_error(self, mock_get):
        """
        総件数取得でAPIエラーが発生した場合のテスト。
        """
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.RequestException("API Error")
        mock_get.return_value = mock_response

        count = get_total_data_papers_count(api_key="TEST_KEY", query="TEST_QUERY")
        self.assertEqual(count, 0)

    @patch('requests.get')
    @patch('time.sleep', return_value=None) # sleepをモック化
    def test_collect_data_papers_success(self, mock_sleep, mock_get):
        """
        データ論文収集が成功した場合のテスト。
        """
        # 最初のレスポンス
        mock_response1 = MagicMock()
        mock_response1.raise_for_status.return_value = None
        mock_response1.json.return_value = {
            'search-results': {
                'entry': [
                    {'eid': 'eid1', 'prism:doi': 'doi1', 'dc:title': 'title1', 'prism:coverDate': '2020-01-01', 'citedby-count': '10'},
                    {'eid': 'eid2', 'prism:doi': 'doi2', 'dc:title': 'title2', 'prism:coverDate': '2021-01-01', 'citedby-count': '20'}
                ],
                'link': [{'@ref': 'next', '@href': 'http://example.com?cursor=next_cursor'}]
            }
        }

        # 2番目のレスポンス (最後のページ)
        mock_response2 = MagicMock()
        mock_response2.raise_for_status.return_value = None
        mock_response2.json.return_value = {
            'search-results': {
                'entry': [
                    {'eid': 'eid3', 'prism:doi': 'doi3', 'dc:title': 'title3', 'prism:coverDate': '2022-01-01', 'citedby-count': '30'}
                ],
                'link': [] # 次のページがないことを示す
            }
        }
        mock_get.side_effect = [mock_response1, mock_response2]

        df = collect_data_papers(api_key="TEST_KEY", query="TEST_QUERY", total_results=3)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.iloc[0]['eid'], 'eid1')
        self.assertEqual(df.iloc[2]['doi'], 'doi3')
        self.assertEqual(mock_get.call_count, 2)

    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_save_data_papers_to_csv_success(self, mock_to_csv, mock_makedirs):
        """
        データ論文の保存が成功した場合のテスト。
        """
        df = pd.DataFrame([
            {'eid': 'eid1', 'doi': 'doi1', 'title': 'title1', 'publication_year': '2020', 'citedby_count': '10'}
        ])
        
        test_output_dir = "test_output"
        test_output_file = os.path.join(test_output_dir, "test_data.csv")

        save_data_papers_to_csv(df, output_file=test_output_file, output_dir=test_output_dir)
        mock_makedirs.assert_called_once_with(test_output_dir, exist_ok=True)
        mock_to_csv.assert_called_once_with(test_output_file, index=False, encoding='utf-8-sig')

    def test_save_data_papers_to_csv_empty_df(self):
        """
        空のDataFrameを保存しようとした場合のテスト。
        """
        df = pd.DataFrame()
        with patch('os.makedirs') as mock_makedirs, \
             patch('pandas.DataFrame.to_csv') as mock_to_csv:
            save_data_papers_to_csv(df)
            mock_makedirs.assert_not_called()
            mock_to_csv.assert_not_called()

if __name__ == '__main__':
    unittest.main()
