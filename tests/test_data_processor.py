import unittest
import pandas as pd
import os
import xml.etree.ElementTree as ET
from unittest.mock import patch, MagicMock

from src.data_processor import extract_text_from_xml_files
from src.config import (
    OUTPUT_FILE_ANNOTATION_TARGET_LIST,
    OUTPUT_FILE_CITING_PAPERS_WITH_PATHS,
    OUTPUT_DIR_PROCESSED,
    OUTPUT_FILE_SAMPLES_WITH_TEXT
)
from src.text_extractor import NAMESPACES # For dummy XML content

class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        self.test_annotation_list_csv = "test_annotation_target_list.csv"
        self.test_master_list_csv = "test_citing_papers_with_paths.csv"
        self.test_output_dir = "test_processed_data"
        self.test_output_file = os.path.join(self.test_output_dir, "test_samples_with_text.csv")
        os.makedirs(self.test_output_dir, exist_ok=True)

        # ダミーXMLファイル
        self.dummy_xml_path_1 = os.path.join(self.test_output_dir, "dummy_article_1.xml")
        self.dummy_xml_path_2 = os.path.join(self.test_output_dir, "dummy_article_2.xml")

        dummy_xml_content_1 = f"""
        <ja:article xmlns:ja="{NAMESPACES['ja']}" xmlns:ce="{NAMESPACES['ce']}" xmlns:dc="{NAMESPACES['dc']}" xmlns:core="{NAMESPACES['core']}" xmlns:xocs="{NAMESPACES['xocs']}">
            <ja:head><ce:abstract><ce:abstract-sec><ce:simple-para>Abstract 1 content.</ce:simple-para></ce:abstract-sec></ce:abstract></ja:head>
            <ja:body><ce:sections><ce:section><ce:section-title>Intro</ce:section-title><ce:para>Full text 1 content.</ce:para></ce:section></ce:sections></ja:body>
        </ja:article>
        """
        dummy_xml_content_2 = f"""
        <ja:article xmlns:ja="{NAMESPACES['ja']}" xmlns:ce="{NAMESPACES['ce']}" xmlns:dc="{NAMESPACES['dc']}" xmlns:core="{NAMESPACES['core']}" xmlns:xocs="{NAMESPACES['xocs']}">
            <ja:head><ce:abstract><ce:abstract-sec><ce:simple-para>Abstract 2 content.</ce:simple-para></ce:abstract-sec></ce:abstract></ja:head>
            <ja:body><ce:sections><ce:section><ce:section-title>Intro</ce:section-title><ce:para>Full text 2 content.</ce:para></ce:section></ce:sections></ja:body>
        </ja:article>
        """
        with open(self.dummy_xml_path_1, "w", encoding="utf-8") as f:
            f.write(dummy_xml_content_1)
        with open(self.dummy_xml_path_2, "w", encoding="utf-8") as f:
            f.write(dummy_xml_content_2)

        # ダミーCSVデータ
        dummy_targets = {
            'citing_paper_eid': ['eid1', 'eid2'],
            'citing_paper_doi': ['doi1', 'doi2'],
            'citing_paper_title': ['Title 1', 'Title 2'],
            'cited_data_paper_title': ['Data Title A', 'Data Title B']
        }
        pd.DataFrame(dummy_targets).to_csv(self.test_annotation_list_csv, index=False)

        dummy_master = {
            'citing_paper_doi': ['doi1', 'doi2', 'doi3'],
            'fulltext_xml_path': [self.dummy_xml_path_1, self.dummy_xml_path_2, 'non_existent.xml'],
            'download_status': ['success', 'success', 'failed']
        }
        pd.DataFrame(dummy_master).to_csv(self.test_master_list_csv, index=False)

    def tearDown(self):
        if os.path.exists(self.test_annotation_list_csv):
            os.remove(self.test_annotation_list_csv)
        if os.path.exists(self.test_master_list_csv):
            os.remove(self.test_master_list_csv)
        if os.path.exists(self.dummy_xml_path_1):
            os.remove(self.dummy_xml_path_1)
        if os.path.exists(self.dummy_xml_path_2):
            os.remove(self.dummy_xml_path_2)
        if os.path.exists(self.test_output_file):
            os.remove(self.test_output_file)
        if os.path.exists(self.test_output_dir):
            os.rmdir(self.test_output_dir)

    @patch('src.text_extractor.extract_abstract_robustly')
    @patch('src.text_extractor.extract_full_text_robustly')
    @patch('os.path.exists') # For xml_path existence checks
    def test_extract_text_from_xml_files_success(self, mock_exists, mock_extract_full_text, mock_extract_abstract):
        mock_exists.side_effect = [True, True, True, True] # All XML files exist
        mock_extract_abstract.side_effect = ['Abstract 1 content.', 'Abstract 2 content.']
        mock_extract_full_text.side_effect = ['Full text 1 content.', 'Full text 2 content.']

        df_extracted = extract_text_from_xml_files(
            annotation_list_csv=self.test_annotation_list_csv,
            master_list_csv=self.test_master_list_csv,
            output_dir=self.test_output_dir,
            output_file_name="test_samples_with_text.csv"
        )
        self.assertFalse(df_extracted.empty)
        self.assertEqual(len(df_extracted), 2)
        self.assertEqual(df_extracted.iloc[0]['abstract'], 'Abstract 1 content.')
        self.assertEqual(df_extracted.iloc[0]['full_text'], 'Full text 1 content.')
        self.assertEqual(df_extracted.iloc[1]['abstract'], 'Abstract 2 content.')
        self.assertEqual(df_extracted.iloc[1]['full_text'], 'Full text 2 content.')
        self.assertTrue(os.path.exists(self.test_output_file))
        self.assertEqual(mock_extract_abstract.call_count, 2)
        self.assertEqual(mock_extract_full_text.call_count, 2)

    @patch('src.text_extractor.extract_abstract_robustly')
    @patch('src.text_extractor.extract_full_text_robustly')
    @patch('os.path.exists') # Simulate one XML file missing
    def test_extract_text_from_xml_files_missing_xml(self, mock_exists, mock_extract_full_text, mock_extract_abstract):
        mock_exists.side_effect = [True, True, True, False] # First XML exists, second does not
        mock_extract_abstract.side_effect = ['Abstract 1 content.', None]
        mock_extract_full_text.side_effect = ['Full text 1 content.', None]

        df_extracted = extract_text_from_xml_files(
            annotation_list_csv=self.test_annotation_list_csv,
            master_list_csv=self.test_master_list_csv,
            output_dir=self.test_output_dir,
            output_file_name="test_samples_with_text.csv"
        )
        self.assertFalse(df_extracted.empty)
        self.assertEqual(len(df_extracted), 2)
        self.assertEqual(df_extracted.iloc[0]['abstract'], 'Abstract 1 content.')
        self.assertIsNone(df_extracted.iloc[1]['abstract']) # Missing XML should result in None
        self.assertTrue(os.path.exists(self.test_output_file))
        # Only called for the existing XML
        self.assertEqual(mock_extract_abstract.call_count, 1) 
        self.assertEqual(mock_extract_full_text.call_count, 1)

    def test_extract_text_from_xml_files_csv_not_found(self):
        df_extracted = extract_text_from_xml_files(
            annotation_list_csv="non_existent_annotation.csv",
            master_list_csv=self.test_master_list_csv,
            output_dir=self.test_output_dir,
            output_file_name="test_samples_with_text.csv"
        )
        self.assertTrue(df_extracted.empty)
        self.assertFalse(os.path.exists(self.test_output_file))

if __name__ == '__main__':
    unittest.main()
