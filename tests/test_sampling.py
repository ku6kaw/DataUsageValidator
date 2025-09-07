import unittest
import pandas as pd
import os
from unittest.mock import patch, MagicMock

from src.sampling import create_annotation_sampling_list
from src.config import OUTPUT_FILE_CITING_PAPERS_WITH_PATHS

class TestSampling(unittest.TestCase):

    def setUp(self):
        self.test_results_csv = "test_citing_papers_with_paths.csv"
        self.test_output_dir = "test_ground_truth"
        self.test_output_file = os.path.join(self.test_output_dir, "test_annotation_target_list.csv")
        os.makedirs(self.test_output_dir, exist_ok=True)

        self.dummy_data = {
            'citing_paper_eid': [f'eid{i}' for i in range(1, 11)],
            'citing_paper_doi': [f'doi{i}' for i in range(1, 11)],
            'citing_paper_title': [f'Title {i}' for i in range(1, 11)],
            'cited_data_paper_title': [f'Data Title {i}' for i in range(1, 11)],
            'download_status': ['success (downloaded)'] * 8 + ['failed'] * 2,
            'fulltext_xml_path': [f'path{i}.xml' for i in range(1, 11)]
        }
        self.df_dummy = pd.DataFrame(self.dummy_data)
        self.df_dummy.to_csv(self.test_results_csv, index=False)

    def tearDown(self):
        if os.path.exists(self.test_results_csv):
            os.remove(self.test_results_csv)
        if os.path.exists(self.test_output_file):
            os.remove(self.test_output_file)
        if os.path.exists(self.test_output_dir):
            os.rmdir(self.test_output_dir)

    def test_create_annotation_sampling_list_success(self):
        sample_df = create_annotation_sampling_list(
            results_csv_path=self.test_results_csv,
            output_dir=self.test_output_dir,
            output_file_name="test_annotation_target_list.csv",
            sample_size=5,
            random_state=42
        )
        self.assertFalse(sample_df.empty)
        self.assertEqual(len(sample_df), 5)
        self.assertTrue(os.path.exists(self.test_output_file))
        
        loaded_df = pd.read_csv(self.test_output_file)
        self.assertEqual(len(loaded_df), 5)
        self.assertListEqual(list(loaded_df.columns), ['citing_paper_eid', 'citing_paper_doi', 'citing_paper_title', 'cited_data_paper_title'])

    def test_create_annotation_sampling_list_file_not_found(self):
        sample_df = create_annotation_sampling_list(
            results_csv_path="non_existent_file.csv",
            output_dir=self.test_output_dir,
            output_file_name="test_annotation_target_list.csv",
            sample_size=5
        )
        self.assertTrue(sample_df.empty)
        self.assertFalse(os.path.exists(self.test_output_file))

    def test_create_annotation_sampling_list_insufficient_samples(self):
        sample_df = create_annotation_sampling_list(
            results_csv_path=self.test_results_csv,
            output_dir=self.test_output_dir,
            output_file_name="test_annotation_target_list.csv",
            sample_size=100 # 成功が8件しかないので足りない
        )
        self.assertTrue(sample_df.empty)
        self.assertFalse(os.path.exists(self.test_output_file))

    def test_create_annotation_sampling_list_no_success_downloads(self):
        # 全て失敗のダミーデータを作成
        dummy_data_failed = {
            'citing_paper_eid': [f'eid{i}' for i in range(1, 5)],
            'citing_paper_doi': [f'doi{i}' for i in range(1, 5)],
            'citing_paper_title': [f'Title {i}' for i in range(1, 5)],
            'cited_data_paper_title': [f'Data Title {i}' for i in range(1, 5)],
            'download_status': ['failed'] * 4,
            'fulltext_xml_path': [f'path{i}.xml' for i in range(1, 5)]
        }
        pd.DataFrame(dummy_data_failed).to_csv(self.test_results_csv, index=False)

        sample_df = create_annotation_sampling_list(
            results_csv_path=self.test_results_csv,
            output_dir=self.test_output_dir,
            output_file_name="test_annotation_target_list.csv",
            sample_size=2
        )
        self.assertTrue(sample_df.empty)
        self.assertFalse(os.path.exists(self.test_output_file))

if __name__ == '__main__':
    unittest.main()
