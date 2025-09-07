import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
from pipeline.collect_citing_papers_pipeline import run_collect_citing_papers_pipeline
from src.config import (
    SCOPUS_API_KEY, OUTPUT_FILE_DATA_PAPERS, OUTPUT_FILE_CITING_PAPERS_WITH_PATHS,
    XML_OUTPUT_DIR
)

class TestCollectCitingPapersPipeline(unittest.TestCase):

    def setUp(self):
        self.test_data_papers_csv = "test_data_papers_for_citing_pipeline.csv"
        self.test_citing_papers_output_csv = "test_citing_papers_with_paths_pipeline.csv"
        self.test_xml_output_dir = "test_xml_output_pipeline"
        os.makedirs(self.test_xml_output_dir, exist_ok=True)

        # Create a dummy data_papers.csv for testing
        dummy_data_papers = {
            'eid': ['eid1', 'eid2'],
            'doi': ['doi1', 'doi2'],
            'title': ['Data Paper 1', 'Data Paper 2'],
            'publication_year': ['2020', '2021'],
            'citedby_count': [15, 5] # eid1 will be processed, eid2 will be filtered out
        }
        pd.DataFrame(dummy_data_papers).to_csv(self.test_data_papers_csv, index=False)

    def tearDown(self):
        if os.path.exists(self.test_data_papers_csv):
            os.remove(self.test_data_papers_csv)
        if os.path.exists(self.test_citing_papers_output_csv):
            os.remove(self.test_citing_papers_output_csv)
        if os.path.exists(os.path.join(self.test_xml_output_dir, 'test_doi_1.xml')):
            os.remove(os.path.join(self.test_xml_output_dir, 'test_doi_1.xml'))
        if os.path.exists(self.test_xml_output_dir):
            os.rmdir(self.test_xml_output_dir)

    @patch('pipeline.collect_citing_papers_pipeline.load_data_papers_for_citing_collection')
    @patch('pipeline.collect_citing_papers_pipeline.list_citing_papers')
    @patch('pipeline.collect_citing_papers_pipeline.download_citing_papers_xml')
    @patch('pipeline.collect_citing_papers_pipeline.save_citing_papers_results')
    @patch('pipeline.collect_citing_papers_pipeline.retry_failed_downloads')
    def test_run_collect_citing_papers_pipeline_success(
        self, mock_retry, mock_save, mock_download_xml, mock_list_citing, mock_load_data
    ):
        """
        引用論文収集パイプラインが正常に実行される場合のテスト。
        """
        # Mock load_data_papers_for_citing_collection
        mock_load_data.return_value = pd.DataFrame({
            'eid': ['eid1'],
            'doi': ['doi1'],
            'title': ['Data Paper 1'],
            'publication_year': ['2020'],
            'citedby_count': [15]
        })

        # Mock list_citing_papers
        mock_list_citing.return_value = [
            {'citing_paper_doi': 'test_doi_1', 'citing_paper_eid': 'ceid1', 'citing_paper_title': 'Citing Paper 1', 'cited_data_paper_title': 'Data Paper 1'}
        ]

        # Mock download_citing_papers_xml
        mock_download_xml.return_value = pd.DataFrame([
            {'citing_paper_doi': 'test_doi_1', 'fulltext_xml_path': 'path/to/xml', 'download_status': 'success'}
        ])

        run_collect_citing_papers_pipeline(
            api_key="TEST_KEY",
            input_data_papers_csv=self.test_data_papers_csv,
            output_citing_papers_csv=self.test_citing_papers_output_csv,
            xml_output_dir=self.test_xml_output_dir,
            min_citations=10,
            max_workers_download_xml=2,
            retry_failed=True
        )

        mock_load_data.assert_called_once_with(input_csv=self.test_data_papers_csv, min_citations=10)
        mock_list_citing.assert_called_once()
        mock_download_xml.assert_called_once()
        mock_save.assert_called_once()
        mock_retry.assert_called_once()

    @patch('pipeline.collect_citing_papers_pipeline.load_data_papers_for_citing_collection')
    @patch('pipeline.collect_citing_papers_pipeline.list_citing_papers')
    @patch('pipeline.collect_citing_papers_pipeline.download_citing_papers_xml')
    @patch('pipeline.collect_citing_papers_pipeline.save_citing_papers_results')
    @patch('pipeline.collect_citing_papers_pipeline.retry_failed_downloads')
    def test_run_collect_citing_papers_pipeline_no_data_papers(
        self, mock_retry, mock_save, mock_download_xml, mock_list_citing, mock_load_data
    ):
        """
        データ論文がない場合、パイプラインがスキップされるテスト。
        """
        mock_load_data.return_value = pd.DataFrame() # 空のDataFrameを返す

        run_collect_citing_papers_pipeline(
            api_key="TEST_KEY",
            input_data_papers_csv=self.test_data_papers_csv,
            output_citing_papers_csv=self.test_citing_papers_output_csv,
            xml_output_dir=self.test_xml_output_dir,
            min_citations=10,
            max_workers_download_xml=2,
            retry_failed=True
        )

        mock_load_data.assert_called_once()
        mock_list_citing.assert_not_called()
        mock_download_xml.assert_not_called()
        mock_save.assert_not_called()
        mock_retry.assert_not_called()

    @patch('pipeline.collect_citing_papers_pipeline.load_data_papers_for_citing_collection')
    @patch('pipeline.collect_citing_papers_pipeline.list_citing_papers')
    @patch('pipeline.collect_citing_papers_pipeline.download_citing_papers_xml')
    @patch('pipeline.collect_citing_papers_pipeline.save_citing_papers_results')
    @patch('pipeline.collect_citing_papers_pipeline.retry_failed_downloads')
    def test_run_collect_citing_papers_pipeline_no_citing_papers(
        self, mock_retry, mock_save, mock_download_xml, mock_list_citing, mock_load_data
    ):
        """
        引用論文がリストアップされない場合、XMLダウンロードがスキップされるテスト。
        """
        mock_load_data.return_value = pd.DataFrame({
            'eid': ['eid1'],
            'doi': ['doi1'],
            'title': ['Data Paper 1'],
            'citedby_count': [15]
        })
        mock_list_citing.return_value = [] # 空のリストを返す

        run_collect_citing_papers_pipeline(
            api_key="TEST_KEY",
            input_data_papers_csv=self.test_data_papers_csv,
            output_citing_papers_csv=self.test_citing_papers_output_csv,
            xml_output_dir=self.test_xml_output_dir,
            min_citations=10,
            max_workers_download_xml=2,
            retry_failed=True
        )

        mock_load_data.assert_called_once()
        mock_list_citing.assert_called_once()
        mock_download_xml.assert_not_called()
        mock_save.assert_not_called()
        mock_retry.assert_not_called()

    @patch('pipeline.collect_citing_papers_pipeline.load_data_papers_for_citing_collection')
    @patch('pipeline.collect_citing_papers_pipeline.list_citing_papers')
    @patch('pipeline.collect_citing_papers_pipeline.download_citing_papers_xml')
    @patch('pipeline.collect_citing_papers_pipeline.save_citing_papers_results')
    @patch('pipeline.collect_citing_papers_pipeline.retry_failed_downloads')
    def test_run_collect_citing_papers_pipeline_no_retry(
        self, mock_retry, mock_save, mock_download_xml, mock_list_citing, mock_load_data
    ):
        """
        retry_failed=Falseの場合、再試行がスキップされるテスト。
        """
        mock_load_data.return_value = pd.DataFrame({
            'eid': ['eid1'],
            'doi': ['doi1'],
            'title': ['Data Paper 1'],
            'citedby_count': [15]
        })
        mock_list_citing.return_value = [
            {'citing_paper_doi': 'test_doi_1', 'citing_paper_eid': 'ceid1', 'citing_paper_title': 'Citing Paper 1', 'cited_data_paper_title': 'Data Paper 1'}
        ]
        mock_download_xml.return_value = pd.DataFrame([
            {'citing_paper_doi': 'test_doi_1', 'fulltext_xml_path': 'path/to/xml', 'download_status': 'success'}
        ])

        run_collect_citing_papers_pipeline(
            api_key="TEST_KEY",
            input_data_papers_csv=self.test_data_papers_csv,
            output_citing_papers_csv=self.test_citing_papers_output_csv,
            xml_output_dir=self.test_xml_output_dir,
            min_citations=10,
            max_workers_download_xml=2,
            retry_failed=False # ここをFalseに設定
        )

        mock_load_data.assert_called_once()
        mock_list_citing.assert_called_once()
        mock_download_xml.assert_called_once()
        mock_save.assert_called_once()
        mock_retry.assert_not_called() # 再試行が呼ばれないことを確認

if __name__ == '__main__':
    unittest.main()
