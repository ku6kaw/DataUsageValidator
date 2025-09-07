import unittest
import xml.etree.ElementTree as ET
import pandas as pd
import os
from unittest.mock import patch, MagicMock

from src.xml_processor import (
    get_citation_map_et,
    find_target_ref_id,
    parse_sections_recursive,
    analyze_single_xml,
    process_xml_for_features,
    namespaces
)
from src.config import (
    OUTPUT_FILE_ANNOTATION_TARGET_LIST,
    OUTPUT_FILE_CITING_PAPERS_WITH_PATHS,
    OUTPUT_DIR_PROCESSED,
    OUTPUT_FILE_FEATURES_FOR_EVALUATION
)

class TestXmlProcessor(unittest.TestCase):

    def setUp(self):
        self.test_xml_path = "test_article.xml"
        self.test_output_dir = "test_processed"
        self.test_features_csv = os.path.join(self.test_output_dir, "test_features.csv")
        os.makedirs(self.test_output_dir, exist_ok=True)

        # ダミーXMLコンテンツ
        self.dummy_xml_content = f"""
        <ja:article xmlns:ja="{namespaces['ja']}" xmlns:ce="{namespaces['ce']}">
            <ja:head>
                <ce:bibliography>
                    <ce:bibliography-sec>
                        <ce:bib-reference id="ref1">
                            <ce:source-text>Citation for Data Paper 1</ce:source-text>
                        </ce:bib-reference>
                        <ce:bib-reference id="ref2">
                            <ce:source-text>Another Citation</ce:source-text>
                        </ce:bib-reference>
                        <ce:bib-reference id="ref3">
                            <ce:source-text>Citation for Target Data Paper</ce:source-text>
                        </ce:bib-reference>
                    </ce:bibliography-sec>
                </ce:bibliography>
            </ja:head>
            <ja:body>
                <ce:sections>
                    <ce:section>
                        <ce:section-title>Introduction</ce:section-title>
                        <ce:para>This is an introduction. <ce:cross-ref refid="ref1"/></ce:para>
                    </ce:section>
                    <ce:section>
                        <ce:section-title>Methods</ce:section-title>
                        <ce:para>We used a method. <ce:cross-ref refid="ref3"/> and <ce:cross-ref refid="ref3"/></ce:para>
                        <ce:section>
                            <ce:section-title>Data Collection</ce:section-title>
                            <ce:para>Data was collected. <ce:cross-ref refid="ref3"/></ce:para>
                        </ce:section>
                    </ce:section>
                    <ce:section>
                        <ce:section-title>Results</ce:section-title>
                        <ce:para>Results are here. <ce:cross-ref refid="ref2"/></ce:para>
                    </ce:section>
                </ce:sections>
            </ja:body>
        </ja:article>
        """
        with open(self.test_xml_path, "w", encoding="utf-8") as f:
            f.write(self.dummy_xml_content)

        # ダミーのCSVファイル
        self.df_targets_csv = "test_annotation_target_list.csv"
        self.df_master_csv = "test_citing_papers_with_paths.csv"

        dummy_targets = {
            'citing_paper_eid': ['ceid1'],
            'citing_paper_doi': ['cdio1'],
            'citing_paper_title': ['Citing Paper Title'],
            'cited_data_paper_title': ['Target Data Paper']
        }
        pd.DataFrame(dummy_targets).to_csv(self.df_targets_csv, index=False)

        dummy_master = {
            'citing_paper_eid': ['ceid1'],
            'citing_paper_doi': ['cdio1'],
            'citing_paper_title': ['Citing Paper Title'],
            'cited_data_paper_title': ['Target Data Paper'],
            'fulltext_xml_path': [self.test_xml_path],
            'download_status': ['success']
        }
        pd.DataFrame(dummy_master).to_csv(self.df_master_csv, index=False)


    def tearDown(self):
        if os.path.exists(self.test_xml_path):
            os.remove(self.test_xml_path)
        if os.path.exists(self.df_targets_csv):
            os.remove(self.df_targets_csv)
        if os.path.exists(self.df_master_csv):
            os.remove(self.df_master_csv)
        if os.path.exists(self.test_features_csv):
            os.remove(self.test_features_csv)
        if os.path.exists(self.test_output_dir):
            os.rmdir(self.test_output_dir)

    def test_get_citation_map_et(self):
        root = ET.fromstring(self.dummy_xml_content)
        citation_map = get_citation_map_et(root)
        self.assertIn('ref1', citation_map)
        self.assertIn('ref3', citation_map)
        self.assertEqual(citation_map['ref3'], 'Citation for Target Data Paper')

    def test_find_target_ref_id(self):
        root = ET.fromstring(self.dummy_xml_content)
        citation_map = get_citation_map_et(root)
        ref_id = find_target_ref_id(citation_map, "Target Data Paper")
        self.assertEqual(ref_id, 'ref3')
        ref_id_not_found = find_target_ref_id(citation_map, "Non Existent Data")
        self.assertIsNone(ref_id_not_found)

    def test_parse_sections_recursive(self):
        root = ET.fromstring(self.dummy_xml_content)
        top_level_sections = root.find('.//ja:body/ce:sections', namespaces)
        sections_data = parse_sections_recursive(top_level_sections)
        
        self.assertEqual(len(sections_data), 4) # Intro, Methods, Data Collection, Results
        self.assertEqual(sections_data[1]['title'], 'Methods')
        self.assertIn('ref3', sections_data[1]['citations'])
        self.assertEqual(sections_data[2]['title'], 'Data Collection')
        self.assertIn('ref3', sections_data[2]['citations'])

    def test_analyze_single_xml(self):
        mention_count, mentioned_sections, pred1, pred2 = analyze_single_xml(self.test_xml_path, "Target Data Paper")
        self.assertEqual(mention_count, 3)
        self.assertIn('Methods', mentioned_sections)
        self.assertIn('Data Collection', mentioned_sections)
        self.assertEqual(pred1, 1) # mention_count >= 2
        self.assertEqual(pred2, 1) # 'Data Collection' contains 'data' keyword

    def test_analyze_single_xml_no_target_ref(self):
        mention_count, mentioned_sections, pred1, pred2 = analyze_single_xml(self.test_xml_path, "Non Existent Data Paper")
        self.assertEqual(mention_count, 0)
        self.assertEqual(mentioned_sections, [])
        self.assertEqual(pred1, 0)
        self.assertEqual(pred2, 0)

    @patch('os.path.exists', return_value=True)
    @patch('src.xml_processor.analyze_single_xml', return_value=(3, ['Methods'], 1, 1))
    def test_process_xml_for_features(self, mock_analyze_single_xml, mock_exists):
        df_features = process_xml_for_features(
            pd.read_csv(self.df_targets_csv),
            pd.read_csv(self.df_master_csv),
            self.test_output_dir,
            "test_features.csv"
        )
        self.assertFalse(df_features.empty)
        self.assertEqual(len(df_features), 1)
        self.assertEqual(df_features.iloc[0]['mention_count'], 3)
        self.assertEqual(df_features.iloc[0]['prediction_rule1'], 1)
        self.assertEqual(df_features.iloc[0]['prediction_rule2'], 1)
        self.assertTrue(os.path.exists(self.test_features_csv))
        mock_analyze_single_xml.assert_called_once()

    @patch('os.path.exists', return_value=False)
    @patch('src.xml_processor.analyze_single_xml')
    def test_process_xml_for_features_file_not_found(self, mock_analyze_single_xml, mock_exists):
        df_features = process_xml_for_features(
            pd.read_csv(self.df_targets_csv),
            pd.read_csv(self.df_master_csv),
            self.test_output_dir,
            "test_features.csv"
        )
        self.assertFalse(df_features.empty)
        self.assertEqual(df_features.iloc[0]['mention_count'], -1)
        self.assertEqual(df_features.iloc[0]['prediction_rule1'], -1)
        self.assertEqual(df_features.iloc[0]['prediction_rule2'], -1)
        mock_analyze_single_xml.assert_not_called()

if __name__ == '__main__':
    unittest.main()
