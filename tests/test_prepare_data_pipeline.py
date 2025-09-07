import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
from pipeline.prepare_data_pipeline import run_prepare_data_pipeline
from src.config import (
    OUTPUT_FILE_CITING_PAPERS_WITH_PATHS,
    OUTPUT_FILE_ANNOTATION_TARGET_LIST,
    OUTPUT_FILE_SAMPLES_WITH_TEXT,
    OUTPUT_DIR_PROCESSED
)

class TestPrepareDataPipeline(unittest.TestCase):

    def setUp(self):
        self.citing_papers_master_csv = "test_citing_papers_master.csv"
        self.annotation_target_list_csv = "test_annotation_target_list.csv"
        self.samples_with_text_csv = "test_samples_with_text.csv"
        self.processed_output_dir = "test_processed_data"
        
        os.makedirs(self.processed_output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.annotation_target_list_csv), exist_ok=True)

        # Create dummy master CSV
        dummy_master_data = {
            'citing_paper_eid': [f'eid{i}' for i in range(1, 6)],
            'citing_paper_doi': [f'doi{i}' for i in range(1, 6)],
            'fulltext_xml_path': [f'path/to/xml{i}.xml' for i in range(1, 6)],
            'download_status': ['success'] * 5
        }
        pd.DataFrame(dummy_master_data).to_csv(self.citing_papers_master_csv, index=False)

    def tearDown(self):
        if os.path.exists(self.citing_papers_master_csv):
            os.remove(self.citing_papers_master_csv)
        if os.path.exists(self.annotation_target_list_csv):
            os.remove(self.annotation_target_list_csv)
        if os.path.exists(self.samples_with_text_csv):
            os.remove(self.samples_with_text_csv)
        
        if os.path.exists(self.processed_output_dir):
            for f in os.listdir(self.processed_output_dir):
                os.remove(os.path.join(self.processed_output_dir, f))
            os.rmdir(self.processed_output_dir)
        
        if os.path.exists(os.path.dirname(self.annotation_target_list_csv)):
            if not os.listdir(os.path.dirname(self.annotation_target_list_csv)): # Check if directory is empty
                os.rmdir(os.path.dirname(self.annotation_target_list_csv))

    @patch('pipeline.prepare_data_pipeline.create_annotation_sampling_list')
    @patch('pipeline.prepare_data_pipeline.extract_text_from_xml_files')
    def test_run_prepare_data_pipeline_success(self, mock_extract_text, mock_create_sampling_list):
        """
        データ準備パイプラインが正常に実行される場合のテスト。
        """
        # Mock create_annotation_sampling_list
        mock_create_sampling_list.return_value = pd.DataFrame({
            'citing_paper_eid': ['eid1', 'eid2'],
            'citing_paper_doi': ['doi1', 'doi2'],
            'citing_paper_title': ['Title 1', 'Title 2'],
            'cited_data_paper_title': ['Data Title 1', 'Data Title 2']
        })

        # Mock extract_text_from_xml_files
        mock_extract_text.return_value = pd.DataFrame({
            'citing_paper_eid': ['eid1', 'eid2'],
            'citing_paper_doi': ['doi1', 'doi2'],
            'citing_paper_title': ['Title 1', 'Title 2'],
            'cited_data_paper_title': ['Data Title 1', 'Data Title 2'],
            'abstract': ['Abstract 1', 'Abstract 2'],
            'full_text': ['Full Text 1', 'Full Text 2']
        })

        run_prepare_data_pipeline(
            citing_papers_master_csv=self.citing_papers_master_csv,
            annotation_target_list_csv=self.annotation_target_list_csv,
            samples_with_text_csv=self.samples_with_text_csv,
            processed_output_dir=self.processed_output_dir,
            sample_size=2,
            random_state=42
        )

        mock_create_sampling_list.assert_called_once()
        mock_extract_text.assert_called_once()

    @patch('pipeline.prepare_data_pipeline.create_annotation_sampling_list')
    @patch('pipeline.prepare_data_pipeline.extract_text_from_xml_files')
    def test_run_prepare_data_pipeline_no_annotation_targets(self, mock_extract_text, mock_create_sampling_list):
        """
        アノテーション対象の論文がない場合、テキスト抽出がスキップされるテスト。
        """
        mock_create_sampling_list.return_value = pd.DataFrame() # 空のDataFrameを返す

        run_prepare_data_pipeline(
            citing_papers_master_csv=self.citing_papers_master_csv,
            annotation_target_list_csv=self.annotation_target_list_csv,
            samples_with_text_csv=self.samples_with_text_csv,
            processed_output_dir=self.processed_output_dir,
            sample_size=2,
            random_state=42
        )

        mock_create_sampling_list.assert_called_once()
        mock_extract_text.assert_not_called()

    @patch('pipeline.prepare_data_pipeline.create_annotation_sampling_list')
    @patch('pipeline.prepare_data_pipeline.extract_text_from_xml_files')
    def test_run_prepare_data_pipeline_empty_extracted_text(self, mock_extract_text, mock_create_sampling_list):
        """
        テキスト抽出結果が空の場合のテスト。
        """
        mock_create_sampling_list.return_value = pd.DataFrame({
            'citing_paper_eid': ['eid1'],
            'citing_paper_doi': ['doi1'],
            'citing_paper_title': ['Title 1'],
            'cited_data_paper_title': ['Data Title 1']
        })
        mock_extract_text.return_value = pd.DataFrame() # 空のDataFrameを返す

        run_prepare_data_pipeline(
            citing_papers_master_csv=self.citing_papers_master_csv,
            annotation_target_list_csv=self.annotation_target_list_csv,
            samples_with_text_csv=self.samples_with_text_csv,
            processed_output_dir=self.processed_output_dir,
            sample_size=1,
            random_state=42
        )

        mock_create_sampling_list.assert_called_once()
        mock_extract_text.assert_called_once()

if __name__ == '__main__':
    unittest.main()
