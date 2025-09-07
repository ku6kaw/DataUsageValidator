import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
from pipeline.review_and_correct_pipeline import run_review_and_correction_pipeline
from src.config import (
    OUTPUT_FILE_ANNOTATION_TARGET_LIST,
    OUTPUT_FILE_PREDICTION_LLM,
    OUTPUT_FILE_SAMPLES_WITH_TEXT
)

class TestReviewAndCorrectPipeline(unittest.TestCase):

    def setUp(self):
        self.ground_truth_csv = "test_ground_truth_pipeline.csv"
        self.llm_predictions_csv = "test_llm_predictions_pipeline.csv"
        self.samples_with_text_csv = "test_samples_with_text_pipeline.csv"
        self.best_model_column = 'prediction_rule3_gemini-2_5-flash'

        # Create dummy CSV files for testing
        self.dummy_gt = pd.DataFrame({
            'citing_paper_doi': ['doi1', 'doi2', 'doi3'],
            'citing_paper_title': ['Title A', 'Title B', 'Title C'],
            'is_data_used_gt': [1, 0, 1]
        })
        self.dummy_gt.to_csv(self.ground_truth_csv, index=False)

        self.dummy_llm = pd.DataFrame({
            'citing_paper_doi': ['doi1', 'doi2', 'doi3'],
            'prediction_rule3_gemini-2_5-flash': [1, 1, 0] # doi2, doi3 are disagreements
        })
        self.dummy_llm.to_csv(self.llm_predictions_csv, index=False)

        self.dummy_samples = pd.DataFrame({
            'citing_paper_doi': ['doi1', 'doi2', 'doi3'],
            'cited_data_paper_title': ['Data A', 'Data B', 'Data C'],
            'citing_paper_title': ['Title A', 'Title B', 'Title C'],
            'full_text': ['Full text for A', 'Full text for B', 'Full text for C']
        })
        self.dummy_samples.to_csv(self.samples_with_text_csv, index=False)

    def tearDown(self):
        if os.path.exists(self.ground_truth_csv):
            os.remove(self.ground_truth_csv)
        if os.path.exists(self.llm_predictions_csv):
            os.remove(self.llm_predictions_csv)
        if os.path.exists(self.samples_with_text_csv):
            os.remove(self.samples_with_text_csv)

    @patch('pipeline.review_and_correct_pipeline.load_and_merge_review_data')
    @patch('pipeline.review_and_correct_pipeline.identify_disagreements')
    @patch('pipeline.review_and_correct_pipeline.generate_review_prompts')
    @patch('pipeline.review_and_correct_pipeline.apply_corrections_to_ground_truth')
    def test_run_review_and_correction_pipeline_success_with_prompts(
        self, mock_apply_corrections, mock_generate_prompts, mock_identify_disagreements, mock_load_and_merge
    ):
        """
        レビューと修正パイプラインがプロンプト生成ありで正常に実行される場合のテスト。
        """
        mock_df_review = pd.DataFrame({
            'citing_paper_doi': ['doi1', 'doi2', 'doi3'],
            'is_data_used_gt': [1, 0, 1],
            'prediction_rule3_gemini-2_5-flash': [1, 1, 0],
            'full_text': ['text1', 'text2', 'text3']
        })
        mock_load_and_merge.return_value = mock_df_review

        mock_disagreements = pd.DataFrame({
            'citing_paper_doi': ['doi2', 'doi3'],
            'citing_paper_title': ['Title B', 'Title C'],
            'Human_Label': ['Not Used', 'Used'],
            'LLM_Prediction': ['Used', 'Not Used']
        })
        mock_identify_disagreements.return_value = mock_disagreements

        run_review_and_correction_pipeline(
            ground_truth_csv=self.ground_truth_csv,
            llm_predictions_csv=self.llm_predictions_csv,
            samples_with_text_csv=self.samples_with_text_csv,
            best_model_column=self.best_model_column,
            corrections=None,
            generate_prompts=True
        )

        mock_load_and_merge.assert_called_once_with(
            ground_truth_csv=self.ground_truth_csv,
            llm_predictions_csv=self.llm_predictions_csv,
            samples_with_text_csv=self.samples_with_text_csv,
            best_model_column=self.best_model_column
        )
        mock_identify_disagreements.assert_called_once_with(mock_df_review, self.best_model_column)
        mock_generate_prompts.assert_called_once_with(mock_disagreements, self.best_model_column)
        mock_apply_corrections.assert_not_called()

    @patch('pipeline.review_and_correct_pipeline.load_and_merge_review_data')
    @patch('pipeline.review_and_correct_pipeline.identify_disagreements')
    @patch('pipeline.review_and_correct_pipeline.generate_review_prompts')
    @patch('pipeline.review_and_correct_pipeline.apply_corrections_to_ground_truth')
    def test_run_review_and_correction_pipeline_success_with_corrections(
        self, mock_apply_corrections, mock_generate_prompts, mock_identify_disagreements, mock_load_and_merge
    ):
        """
        レビューと修正パイプラインが修正適用ありで正常に実行される場合のテスト。
        """
        mock_df_review = pd.DataFrame({
            'citing_paper_doi': ['doi1', 'doi2', 'doi3'],
            'is_data_used_gt': [1, 0, 1],
            'prediction_rule3_gemini-2_5-flash': [1, 1, 0],
            'full_text': ['text1', 'text2', 'text3']
        })
        mock_load_and_merge.return_value = mock_df_review

        mock_disagreements = pd.DataFrame({
            'citing_paper_doi': ['doi2', 'doi3'],
            'citing_paper_title': ['Title B', 'Title C'],
            'Human_Label': ['Not Used', 'Used'],
            'LLM_Prediction': ['Used', 'Not Used']
        })
        mock_identify_disagreements.return_value = mock_disagreements

        test_corrections = {'doi2': 0}

        run_review_and_correction_pipeline(
            ground_truth_csv=self.ground_truth_csv,
            llm_predictions_csv=self.llm_predictions_csv,
            samples_with_text_csv=self.samples_with_text_csv,
            best_model_column=self.best_model_column,
            corrections=test_corrections,
            generate_prompts=False # プロンプト生成はスキップ
        )

        mock_load_and_merge.assert_called_once()
        mock_identify_disagreements.assert_called_once()
        mock_generate_prompts.assert_not_called()
        mock_apply_corrections.assert_called_once_with(test_corrections, self.ground_truth_csv)

    @patch('pipeline.review_and_correct_pipeline.load_and_merge_review_data')
    @patch('pipeline.review_and_correct_pipeline.identify_disagreements')
    @patch('pipeline.review_and_correct_pipeline.generate_review_prompts')
    @patch('pipeline.review_and_correct_pipeline.apply_corrections_to_ground_truth')
    def test_run_review_and_correction_pipeline_no_review_data(
        self, mock_apply_corrections, mock_generate_prompts, mock_identify_disagreements, mock_load_and_merge
    ):
        """
        レビュー対象のデータがない場合、パイプラインがスキップされるテスト。
        """
        mock_load_and_merge.return_value = pd.DataFrame() # 空のDataFrameを返す

        run_review_and_correction_pipeline(
            ground_truth_csv=self.ground_truth_csv,
            llm_predictions_csv=self.llm_predictions_csv,
            samples_with_text_csv=self.samples_with_text_csv,
            best_model_column=self.best_model_column,
            corrections=None,
            generate_prompts=True
        )

        mock_load_and_merge.assert_called_once()
        mock_identify_disagreements.assert_not_called()
        mock_generate_prompts.assert_not_called()
        mock_apply_corrections.assert_not_called()

if __name__ == '__main__':
    unittest.main()
