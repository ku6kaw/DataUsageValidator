import unittest
import pandas as pd
import os
from unittest.mock import patch, MagicMock
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.evaluation import (
    load_and_merge_evaluation_data,
    generate_hybrid_predictions,
    calculate_metrics,
    save_evaluation_results
)
from src.config import (
    OUTPUT_FILE_ANNOTATION_TARGET_LIST,
    OUTPUT_FILE_FEATURES_FOR_EVALUATION,
    OUTPUT_FILE_PREDICTION_LLM,
    TABLES_DIR
)

class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.gt_csv = "test_ground_truth.csv"
        self.features_csv = "test_features.csv"
        self.llm_csv = "test_llm_predictions.csv"
        self.output_dir = "test_results_tables"
        os.makedirs(self.output_dir, exist_ok=True)

        # ダミー正解データ
        self.dummy_gt = pd.DataFrame({
            'citing_paper_doi': ['doi1', 'doi2', 'doi3', 'doi4', 'doi5'],
            'is_data_used_gt': [1, 0, 1, 0, 1]
        })
        self.dummy_gt.to_csv(self.gt_csv, index=False)

        # ダミー特徴量データ
        self.dummy_features = pd.DataFrame({
            'citing_paper_doi': ['doi1', 'doi2', 'doi3', 'doi4', 'doi5'],
            'prediction_rule1': [1, 0, 1, 1, 0],
            'prediction_rule2': [1, 0, 0, 1, 1]
        })
        self.dummy_features.to_csv(self.features_csv, index=False)

        # ダミーLLM予測データ
        self.dummy_llm = pd.DataFrame({
            'citing_paper_doi': ['doi1', 'doi2', 'doi3', 'doi4', 'doi5'],
            'prediction_rule3_abstract': [1, 0, 1, 0, 1],
            'prediction_rule3_fulltext': [1, 0, 0, 0, 1],
            'prediction_rule3_fulltext_few_shot': [1, 0, 1, 0, 1],
            'prediction_rule3_gemini-2_5-flash': [1, 0, 1, 0, 1],
            'prediction_rule3_gemini-2_5-flash_zeroshot': [1, 0, 0, 0, 1]
        })
        self.dummy_llm.to_csv(self.llm_csv, index=False)

    def tearDown(self):
        if os.path.exists(self.gt_csv):
            os.remove(self.gt_csv)
        if os.path.exists(self.features_csv):
            os.remove(self.features_csv)
        if os.path.exists(self.llm_csv):
            os.remove(self.llm_csv)
        if os.path.exists(os.path.join(self.output_dir, 'evaluation_metrics_summary.csv')):
            os.remove(os.path.join(self.output_dir, 'evaluation_metrics_summary.csv'))
        if os.path.exists(self.output_dir):
            os.rmdir(self.output_dir)

    def test_load_and_merge_evaluation_data_success(self):
        df_eval = load_and_merge_evaluation_data(self.gt_csv, self.features_csv, self.llm_csv)
        self.assertFalse(df_eval.empty)
        self.assertEqual(len(df_eval), 5)
        self.assertIn('is_data_used_gt', df_eval.columns)
        self.assertIn('prediction_rule1', df_eval.columns)
        self.assertIn('prediction_rule3_fulltext', df_eval.columns)

    def test_load_and_merge_evaluation_data_file_not_found(self):
        df_eval = load_and_merge_evaluation_data("non_existent.csv", self.features_csv, self.llm_csv)
        self.assertTrue(df_eval.empty)

    def test_generate_hybrid_predictions(self):
        df_eval_base = load_and_merge_evaluation_data(self.gt_csv, self.features_csv, self.llm_csv)
        df_hybrid = generate_hybrid_predictions(df_eval_base)
        self.assertIn('prediction_hybrid_AND_zeroshot', df_hybrid.columns)
        self.assertIn('prediction_hierarchical_hybrid', df_hybrid.columns)
        
        # 例: doi1 (Rule2=1, LLM_fulltext=1) -> hybrid_AND_zeroshot = 1
        self.assertEqual(df_hybrid.loc[df_hybrid['citing_paper_doi'] == 'doi1', 'prediction_hybrid_AND_zeroshot'].iloc[0], 1)
        # 例: doi3 (Rule2=0, LLM_fulltext=0) -> hybrid_AND_zeroshot = 0
        self.assertEqual(df_hybrid.loc[df_hybrid['citing_paper_doi'] == 'doi3', 'prediction_hybrid_AND_zeroshot'].iloc[0], 0)

    def test_calculate_metrics(self):
        df_eval_base = load_and_merge_evaluation_data(self.gt_csv, self.features_csv, self.llm_csv)
        df_hybrid = generate_hybrid_predictions(df_eval_base)
        
        rule_columns = {
            "Rule 1": "prediction_rule1",
            "Hybrid": "prediction_hybrid_AND_zeroshot"
        }
        df_metrics = calculate_metrics(df_hybrid, rule_columns)
        self.assertFalse(df_metrics.empty)
        self.assertEqual(len(df_metrics), 2)
        self.assertIn('F1-Score', df_metrics.columns)
        
        # Rule 1 のF1スコアを検証 (手計算: TP=3, TN=1, FP=1, FN=1 -> P=3/4=0.75, R=3/4=0.75, F1=0.75)
        rule1_f1 = df_metrics.loc[df_metrics['Rule'] == 'Rule 1', 'F1-Score'].iloc[0]
        self.assertAlmostEqual(rule1_f1, 0.75, places=2)

    @patch('builtins.print')
    def test_save_evaluation_results(self, mock_print):
        df_metrics = pd.DataFrame({
            'Rule': ['Test Rule'],
            'TP': [1], 'TN': [1], 'FP': [1], 'FN': [1],
            'Accuracy': [0.5], 'Precision': [0.5], 'Recall': [0.5], 'F1-Score': [0.5],
            'Eval_Count': [4]
        })
        save_evaluation_results(df_metrics, output_file_name='test_metrics.csv')
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'test_metrics.csv')))
        mock_print.assert_called()

if __name__ == '__main__':
    unittest.main()
