import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
from pipeline.evaluate_results_pipeline import run_evaluate_results_pipeline
from src.config import (
    OUTPUT_FILE_ANNOTATION_TARGET_LIST,
    OUTPUT_FILE_FEATURES_FOR_EVALUATION,
    OUTPUT_FILE_PREDICTION_LLM,
    TABLES_DIR
)

class TestEvaluateResultsPipeline(unittest.TestCase):

    def setUp(self):
        self.ground_truth_csv = "test_ground_truth_pipeline.csv"
        self.features_csv = "test_features_pipeline.csv"
        self.llm_predictions_csv = "test_llm_predictions_pipeline.csv"
        self.output_metrics_file_name = "test_evaluation_metrics_summary.csv"
        self.test_tables_dir = "test_results_tables_pipeline"
        
        os.makedirs(self.test_tables_dir, exist_ok=True)

        # Create dummy CSV files for testing
        self.dummy_gt = pd.DataFrame({
            'citing_paper_doi': ['doi1', 'doi2'],
            'is_data_used_gt': [1, 0]
        })
        self.dummy_gt.to_csv(self.ground_truth_csv, index=False)

        self.dummy_features = pd.DataFrame({
            'citing_paper_doi': ['doi1', 'doi2'],
            'prediction_rule1': [1, 0],
            'prediction_rule2': [1, 1]
        })
        self.dummy_features.to_csv(self.features_csv, index=False)

        self.dummy_llm = pd.DataFrame({
            'citing_paper_doi': ['doi1', 'doi2'],
            'prediction_rule3_abstract': [1, 0],
            'prediction_rule3_fulltext': [1, 1],
            'prediction_rule3_gemini-2_5-flash_zeroshot': [1, 1]
        })
        self.dummy_llm.to_csv(self.llm_predictions_csv, index=False)

    def tearDown(self):
        if os.path.exists(self.ground_truth_csv):
            os.remove(self.ground_truth_csv)
        if os.path.exists(self.features_csv):
            os.remove(self.features_csv)
        if os.path.exists(self.llm_predictions_csv):
            os.remove(self.llm_predictions_csv)
        
        output_path = os.path.join(self.test_tables_dir, self.output_metrics_file_name)
        if os.path.exists(output_path):
            os.remove(output_path)
        if os.path.exists(self.test_tables_dir):
            os.rmdir(self.test_tables_dir)

    @patch('pipeline.evaluate_results_pipeline.load_and_merge_evaluation_data')
    @patch('pipeline.evaluate_results_pipeline.generate_hybrid_predictions')
    @patch('pipeline.evaluate_results_pipeline.calculate_metrics')
    @patch('pipeline.evaluate_results_pipeline.save_evaluation_results')
    def test_run_evaluate_results_pipeline_success(
        self, mock_save_results, mock_calculate_metrics, mock_generate_hybrid, mock_load_and_merge
    ):
        """
        評価と分析パイプラインが正常に実行される場合のテスト。
        """
        mock_df_eval_base = pd.DataFrame({
            'citing_paper_doi': ['doi1', 'doi2'],
            'is_data_used_gt': [1, 0],
            'prediction_rule1': [1, 0],
            'prediction_rule2': [1, 1],
            'prediction_rule3_abstract': [1, 0]
        })
        mock_load_and_merge.return_value = mock_df_eval_base

        mock_df_hybrid = mock_df_eval_base.copy()
        mock_df_hybrid['prediction_hybrid_AND_zeroshot'] = [1, 0]
        mock_generate_hybrid.return_value = mock_df_hybrid

        mock_df_metrics = pd.DataFrame({
            'Rule': ['Rule 1'],
            'F1-Score': [0.8]
        })
        mock_calculate_metrics.return_value = mock_df_metrics

        run_evaluate_results_pipeline(
            ground_truth_csv=self.ground_truth_csv,
            features_csv=self.features_csv,
            llm_predictions_csv=self.llm_predictions_csv,
            output_metrics_file_name=self.output_metrics_file_name
        )

        mock_load_and_merge.assert_called_once_with(
            ground_truth_csv=self.ground_truth_csv,
            features_csv=self.features_csv,
            llm_predictions_csv=self.llm_predictions_csv
        )
        mock_generate_hybrid.assert_called_once_with(mock_df_eval_base)
        mock_calculate_metrics.assert_called_once()
        mock_save_results.assert_called_once_with(mock_df_metrics, self.output_metrics_file_name)

    @patch('pipeline.evaluate_results_pipeline.load_and_merge_evaluation_data')
    @patch('pipeline.evaluate_results_pipeline.generate_hybrid_predictions')
    @patch('pipeline.evaluate_results_pipeline.calculate_metrics')
    @patch('pipeline.evaluate_results_pipeline.save_evaluation_results')
    def test_run_evaluate_results_pipeline_no_eval_data(
        self, mock_save_results, mock_calculate_metrics, mock_generate_hybrid, mock_load_and_merge
    ):
        """
        評価対象のデータがない場合、パイプラインがスキップされるテスト。
        """
        mock_load_and_merge.return_value = pd.DataFrame() # 空のDataFrameを返す

        run_evaluate_results_pipeline(
            ground_truth_csv=self.ground_truth_csv,
            features_csv=self.features_csv,
            llm_predictions_csv=self.llm_predictions_csv,
            output_metrics_file_name=self.output_metrics_file_name
        )

        mock_load_and_merge.assert_called_once()
        mock_generate_hybrid.assert_not_called()
        mock_calculate_metrics.assert_not_called()
        mock_save_results.assert_not_called()

if __name__ == '__main__':
    unittest.main()
