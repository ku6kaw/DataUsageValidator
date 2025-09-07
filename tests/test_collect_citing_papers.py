import unittest
import pandas as pd
import os
from unittest.mock import patch, MagicMock
import requests

from src.collect_citing_papers import (
    load_data_papers_for_citing_collection,
    list_citing_papers,
    sanitize_filename,
    download_xml_by_doi,
    download_citing_papers_xml,
    save_citing_papers_results,
    retry_failed_downloads
)
from src.config import (
    OUTPUT_FILE_DATA_PAPERS,
    OUTPUT_FILE_CITING_PAPERS_WITH_PATHS,
    XML_OUTPUT_DIR,
    SCOPUS_BASE_URL,
    SCOPUS_FULLTEXT_API_DOI_URL
)

class TestCollectCitingPapers(unittest.TestCase):

    def setUp(self):
        self.test_data_papers_csv = "test_data_papers_for_citing.csv"
        self.test_citing_papers_raw_csv = "test_citing_papers_raw.csv"
        self.test_citing_papers_with_paths_csv = "test_citing_papers_with_paths.csv"
        self.test_xml_dir = "test_xml_output"
        os.makedirs(self.test_xml_dir, exist_ok=True)

        # Dummy data_papers.csv
        dummy_data_papers = {
            'eid': ['eid1', 'eid2'],
            'doi': ['doi1', 'doi2'],
            'title': ['Data Paper 1', 'Data Paper 2'],
            'publication_year': ['2020', '2021'],
            'citedby_count': [15, 5]
        }
        pd.DataFrame(dummy_data_papers).to_csv(self.test_data_papers_csv, index=False)

    def tearDown(self):
        if os.path.exists(self.test_data_papers_csv):
            os.remove(self.test_data_papers_csv)
        if os.path.exists(self.test_citing_papers_raw_csv):
            os.remove(self.test_citing_papers_raw_csv)
        if os.path.exists(self.test_citing_papers_with_paths_csv):
            os.remove(self.test_citing_papers_with_paths_csv)
        if os.path.exists(os.path.join(self.test_xml_dir, 'test_doi_1.xml')):
            os.remove(os.path.join(self.test_xml_dir, 'test_doi_1.xml'))
        if os.path.exists(os.path.join(self.test_xml_dir, 'test_doi_2.xml')):
            os.remove(os.path.join(self.test_xml_dir, 'test_doi_2.xml'))
        if os.path.exists(self.test_xml_dir):
            os.rmdir(self.test_xml_dir)

    def test_load_data_papers_for_citing_collection(self):
        df = load_data_papers_for_citing_collection(self.test_data_papers_csv, min_citations=10)
        self.assertEqual(len(df), 1) # Only 'eid1' has citedby_count >= 10
        self.assertEqual(df.iloc[0]['eid'], 'eid1')

    @patch('requests.get')
    @patch('time.sleep', return_value=None)
    def test_list_citing_papers(self, mock_sleep, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'search-results': {
                'entry': [
                    {'eid': 'ceid1', 'prism:doi': 'cdio1', 'dc:title': 'Citing Paper 1', 'prism:coverDate': '2022-01-01'},
                ],
                'link': []
            }
        }
        mock_get.return_value = mock_response

        df_data_papers = load_data_papers_for_citing_collection(self.test_data_papers_csv, min_citations=10)
        tasks = list_citing_papers(df_data_papers, api_key="TEST_KEY")
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]['citing_paper_eid'], 'ceid1')
        mock_get.assert_called_once()

    def test_sanitize_filename(self):
        self.assertEqual(sanitize_filename("10.1000/test/file?name"), "10.1000_test_file_name")
        self.assertEqual(sanitize_filename("simple_name.xml"), "simple_name.xml")

    @patch('requests.get')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('os.path.exists', return_value=False)
    @patch('time.sleep', return_value=None)
    def test_download_xml_by_doi_success(self, mock_sleep, mock_exists, mock_open, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<article>XML Content</article>"
        mock_get.return_value = mock_response

        task = {'citing_paper_doi': 'test_doi_1'}
        result_task = download_xml_by_doi(task, api_key="TEST_KEY", output_dir=self.test_xml_dir)
        self.assertIn('success', result_task['download_status'])
        self.assertIsNotNone(result_task['fulltext_xml_path'])
        mock_get.assert_called_once()
        mock_open.assert_called_once()

    @patch('requests.get')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('os.path.exists', return_value=True)
    def test_download_xml_by_doi_cached(self, mock_exists, mock_open, mock_get):
        task = {'citing_paper_doi': 'test_doi_1'}
        result_task = download_xml_by_doi(task, api_key="TEST_KEY", output_dir=self.test_xml_dir)
        self.assertIn('cached', result_task['download_status'])
        self.assertIsNotNone(result_task['fulltext_xml_path'])
        mock_get.assert_not_called()
        mock_open.assert_not_called()

    @patch('requests.get')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('os.path.exists', return_value=False)
    @patch('time.sleep', return_value=None)
    def test_download_xml_by_doi_429_retry_success(self, mock_sleep, mock_exists, mock_open, mock_get):
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.text = "<article>XML Content</article>"
        mock_get.side_effect = [mock_response_429, mock_response_success]

        task = {'citing_paper_doi': 'test_doi_2'}
        result_task = download_xml_by_doi(task, api_key="TEST_KEY", output_dir=self.test_xml_dir, max_retries=2)
        self.assertIn('success', result_task['download_status'])
        self.assertIsNotNone(result_task['fulltext_xml_path'])
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)

    @patch('src.collect_citing_papers.download_xml_by_doi')
    @patch('os.makedirs', return_value=None)
    def test_download_citing_papers_xml(self, mock_makedirs, mock_download_xml_by_doi):
        mock_download_xml_by_doi.side_effect = [
            {'citing_paper_doi': 'cdio1', 'fulltext_xml_path': 'path1', 'download_status': 'success'},
            {'citing_paper_doi': 'cdio2', 'fulltext_xml_path': 'path2', 'download_status': 'success'}
        ]
        tasks = [
            {'citing_paper_doi': 'cdio1'},
            {'citing_paper_doi': 'cdio2'}
        ]
        df_results = download_citing_papers_xml(tasks, api_key="TEST_KEY", output_dir=self.test_xml_dir, max_workers=2)
        self.assertEqual(len(df_results), 2)
        self.assertEqual(mock_download_xml_by_doi.call_count, 2)
        mock_makedirs.assert_called_once()

    @patch('pandas.DataFrame.to_csv')
    @patch('os.makedirs', return_value=None)
    def test_save_citing_papers_results(self, mock_makedirs, mock_to_csv):
        df = pd.DataFrame([{'citing_paper_doi': 'cdio1', 'download_status': 'success'}])
        save_citing_papers_results(df, output_csv=self.test_citing_papers_with_paths_csv)
        mock_to_csv.assert_called_once_with(self.test_citing_papers_with_paths_csv, index=False, encoding='utf-8-sig')

    @patch('src.collect_citing_papers.download_citing_papers_xml')
    @patch('src.collect_citing_papers.save_citing_papers_results')
    def test_retry_failed_downloads(self, mock_save, mock_download_xml):
        # Create a dummy CSV with failed downloads
        dummy_results = {
            'citing_paper_doi': ['d1', 'd2', 'd3'],
            'download_status': ['success', 'failed (Status: 429)', 'failed (Request Error)'],
            'fulltext_xml_path': ['path1', None, None]
        }
        pd.DataFrame(dummy_results).to_csv(self.test_citing_papers_with_paths_csv, index=False)

        mock_download_xml.return_value = pd.DataFrame({
            'citing_paper_doi': ['d2', 'd3'],
            'download_status': ['success (retry)', 'failed (retries exhausted)'],
            'fulltext_xml_path': ['path2_new', None]
        })

        retry_failed_downloads(input_csv=self.test_citing_papers_with_paths_csv, api_key="TEST_KEY", output_dir=self.test_xml_dir)
        mock_download_xml.assert_called_once()
        mock_save.assert_called_once()

if __name__ == '__main__':
    unittest.main()
