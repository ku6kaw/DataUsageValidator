import unittest
import pandas as pd
import os
from unittest.mock import patch, MagicMock

from src.review_and_correction import (
    load_and_merge_review_data,
    identify_disagreements,
    generate_review_prompts,
    apply_corrections_to_ground_truth
)
from src.config import (
    OUTPUT_FILE_ANNOTATION_TARGET_LIST,
    OUTPUT_FILE_PREDICTION_LLM,
    OUTPUT_FILE_SAMPLES_WITH_TEXT
)

class TestReviewAndCorrection(unittest.TestCase):

    def setUp(self):
        self.gt_csv = "test_ground_truth.csv"
        self.llm_csv = "test_llm_predictions.csv"
        self.samples_csv = "test_samples_with_text.csv"

        # ダミー正解データ
        self.dummy_gt = pd.DataFrame({
            'citing_paper_doi': ['doi1', 'doi2', 'doi3', 'doi4', 'doi5'],
            'citing_paper_title': ['Title A', 'Title B', 'Title C', 'Title D', 'Title E'],
            'is_data_used_gt': [1, 0, 1, 0, 1]
        })
        self.dummy_gt.to_csv(self.gt_csv, index=False)

        # ダミーLLM予測データ
        self.dummy_llm = pd.DataFrame({
            'citing_paper_doi': ['doi1', 'doi2', 'doi3', 'doi4', 'doi5'],
            'prediction_rule3_gemini-2_5-flash': [1, 1, 0, 0, -1] # doi2, doi3, doi5が不一致/エラー
        })
        self.dummy_llm.to_csv(self.llm_csv, index=False)

        # ダミーsamples_with_textデータ
        self.dummy_samples = pd.DataFrame({
            'citing_paper_doi': ['doi1', 'doi2', 'doi3', 'doi4', 'doi5'],
            'cited_data_paper_title': ['Data A', 'Data B', 'Data C', 'Data D', 'Data E'],
            'citing_paper_title': ['Title A', 'Title B', 'Title C', 'Title D', 'Title E'],
            'full_text': ['Full text for A', 'Full text for B', 'Full text for C', 'Full text for D', 'Full text for E']
        })
        self.dummy_samples.to_csv(self.samples_csv, index=False)

    def tearDown(self):
        if os.path.exists(self.gt_csv):
            os.remove(self.gt_csv)
        if os.path.exists(self.llm_csv):
            os.remove(self.llm_csv)
        if os.path.exists(self.samples_csv):
            os.remove(self.samples_csv)

    def test_load_and_merge_review_data_success(self):
        df_review = load_and_merge_review_data(self.gt_csv, self.llm_csv, self.samples_csv)
        self.assertFalse(df_review.empty)
        self.assertEqual(len(df_review), 4) # doi5はLLM予測が-1なので除外される
        self.assertIn('is_data_used_gt', df_review.columns)
        self.assertIn('prediction_rule3_gemini-2_5-flash', df_review.columns)
        self.assertIn('full_text', df_review.columns)

    def test_load_and_merge_review_data_file_not_found(self):
        df_review = load_and_merge_review_data("non_existent.csv", self.llm_csv, self.samples_csv)
        self.assertTrue(df_review.empty)

    def test_identify_disagreements(self):
        df_review = load_and_merge_review_data(self.gt_csv, self.llm_csv, self.samples_csv)
        disagreements = identify_disagreements(df_review, 'prediction_rule3_gemini-2_5-flash')
        self.assertFalse(disagreements.empty)
        self.assertEqual(len(disagreements), 2) # doi2 (GT=0, LLM=1), doi3 (GT=1, LLM=0)
        self.assertEqual(disagreements.iloc[0]['citing_paper_doi'], 'doi2')
        self.assertEqual(disagreements.iloc[0]['Human_Label'], 'Not Used')
        self.assertEqual(disagreements.iloc[0]['LLM_Prediction'], 'Used')

    @patch('builtins.print')
    def test_generate_review_prompts(self, mock_print):
        df_review = load_and_merge_review_data(self.gt_csv, self.llm_csv, self.samples_csv)
        disagreements = identify_disagreements(df_review, 'prediction_rule3_gemini-2_5-flash')
        generate_review_prompts(disagreements)
        mock_print.assert_called()
        # プロンプトの内容の一部がprintされたことを確認
        mock_print.assert_any_call(unittest.mock.ANY)

    def test_apply_corrections_to_ground_truth(self):
        corrections = {
            'doi2': 0, # LLMが1と予測したがGTは0、修正なし
            'doi3': 1, # LLMが0と予測したがGTは1、修正なし
            'doi4': 1  # GTが0だが、これを1に修正
        }
        apply_corrections_to_ground_truth(corrections, self.gt_csv)
        
        loaded_gt = pd.read_csv(self.gt_csv)
        self.assertEqual(loaded_gt.loc[loaded_gt['citing_paper_doi'] == 'doi2', 'is_data_used_gt'].iloc[0], 0)
        self.assertEqual(loaded_gt.loc[loaded_gt['citing_paper_doi'] == 'doi3', 'is_data_used_gt'].iloc[0], 1)
        self.assertEqual(loaded_gt.loc[loaded_gt['citing_paper_doi'] == 'doi4', 'is_data_used_gt'].iloc[0], 1)

    def test_apply_corrections_to_ground_truth_no_corrections(self):
        original_df = pd.read_csv(self.gt_csv)
        apply_corrections_to_ground_truth({}, self.gt_csv)
        loaded_df = pd.read_csv(self.gt_csv)
        pd.testing.assert_frame_equal(original_df, loaded_df) # 変更がないことを確認

if __name__ == '__main__':
    unittest.main()
