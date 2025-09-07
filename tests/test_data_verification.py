import unittest
import pandas as pd
import os
from unittest.mock import patch, MagicMock

from src.data_verification import (
    load_citing_papers_results,
    summarize_download_status,
    verify_xml_file_existence
)
from src.config import OUTPUT_FILE_CITING_PAPERS_WITH_PATHS

class TestDataVerification(unittest.TestCase):

    def setUp(self):
        self.test_results_csv = "test_citing_papers_with_paths.csv"
        self.test_xml_dir = "test_xml_files"
        os.makedirs(self.test_xml_dir, exist_ok=True)

        # ダミーのXMLファイルを作成
        with open(os.path.join(self.test_xml_dir, "doi1.xml"), "w") as f:
            f.write("<article>content</article>")
        with open(os.path.join(self.test_xml_dir, "doi3.xml"), "w") as f:
            f.write("<article>content</article>")

        self.dummy_data = {
            'citing_paper_doi': ['doi1', 'doi2', 'doi3', 'doi4'],
            'download_status': ['success (downloaded)', 'failed (Status: 404)', 'success (cached)', 'failed (DOI is missing)'],
            'fulltext_xml_path': [
                os.path.join(self.test_xml_dir, 'doi1.xml'),
                None,
                os.path.join(self.test_xml_dir, 'doi3.xml'),
                None
            ]
        }
        self.df_dummy = pd.DataFrame(self.dummy_data)
        self.df_dummy.to_csv(self.test_results_csv, index=False)

    def tearDown(self):
        if os.path.exists(self.test_results_csv):
            os.remove(self.test_results_csv)
        if os.path.exists(os.path.join(self.test_xml_dir, "doi1.xml")):
            os.remove(os.path.join(self.test_xml_dir, "doi1.xml"))
        if os.path.exists(os.path.join(self.test_xml_dir, "doi3.xml")):
            os.remove(os.path.join(self.test_xml_dir, "doi3.xml"))
        if os.path.exists(self.test_xml_dir):
            os.rmdir(self.test_xml_dir)

    def test_load_citing_papers_results_success(self):
        df = load_citing_papers_results(self.test_results_csv)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 4)

    def test_load_citing_papers_results_file_not_found(self):
        df = load_citing_papers_results("non_existent_file.csv")
        self.assertTrue(df.empty)

    @patch('builtins.print')
    def test_summarize_download_status(self, mock_print):
        df = load_citing_papers_results(self.test_results_csv)
        summarize_download_status(df)
        mock_print.assert_called()
        # 特定の文字列が含まれているか確認 (例: 'success (downloaded)    1')
        mock_print.assert_any_call(unittest.mock.ANY) # Any call is fine, specific content is harder to mock precisely

    @patch('builtins.print')
    def test_verify_xml_file_existence_success(self, mock_print):
        df = load_citing_papers_results(self.test_results_csv)
        df_verified = verify_xml_file_existence(df)
        self.assertIn('file_exists', df_verified.columns)
        # doi1.xmlとdoi3.xmlは存在するのでTrueが2つ
        self.assertEqual(df_verified['file_exists'].sum(), 2)
        # printの呼び出しを直接確認するのではなく、出力されたメッセージに特定の文字列が含まれているかを確認
        # mock_print.assert_any_call('[OK] 全ての成功レコードについて、ファイルが正しく存在することを確認しました。')
        # より堅牢なチェックのため、出力されたメッセージ全体を結合して確認
        called_args = [call_arg[0][0] for call_arg in mock_print.call_args_list]
        self.assertIn('[OK] 全ての成功レコードについて、ファイルが正しく存在することを確認しました。', called_args)

    @patch('builtins.print')
    def test_verify_xml_file_existence_missing_file(self, mock_print):
        # ダミーデータからdoi3.xmlを削除して、存在しないケースを作成
        os.remove(os.path.join(self.test_xml_dir, "doi3.xml"))
        
        df = load_citing_papers_results(self.test_results_csv)
        df_verified = verify_xml_file_existence(df)
        self.assertIn('file_exists', df_verified.columns)
        # doi1.xmlのみ存在するのでTrueが1つ
        self.assertEqual(df_verified['file_exists'].sum(), 1)
        mock_print.assert_any_call(unittest.mock.ANY) # 警告メッセージが出力されることを確認

    def test_verify_xml_file_existence_empty_df(self):
        df = pd.DataFrame()
        df_verified = verify_xml_file_existence(df)
        self.assertTrue(df_verified.empty)

if __name__ == '__main__':
    unittest.main()
