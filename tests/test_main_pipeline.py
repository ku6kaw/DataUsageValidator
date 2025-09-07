import unittest
from unittest.mock import patch, MagicMock
import argparse
import os
from pipeline.main_pipeline import main_pipeline
from src.config import (
    SCOPUS_API_KEY, GEMINI_API_KEY, GEMINI_MODEL_NAME,
    OUTPUT_FILE_DATA_PAPERS, OUTPUT_FILE_CITING_PAPERS_WITH_PATHS,
    OUTPUT_FILE_ANNOTATION_TARGET_LIST, OUTPUT_FILE_SAMPLES_WITH_TEXT,
    OUTPUT_FILE_PREDICTION_LLM, OUTPUT_FILE_FEATURES_FOR_EVALUATION,
    OUTPUT_DIR_PROCESSED, XML_OUTPUT_DIR
)

class TestMainPipeline(unittest.TestCase):

    def setUp(self):
        # Clean up any existing dummy files/dirs from previous runs
        self._cleanup_files()

        # Create dummy config files/dirs if needed for pipeline imports
        os.makedirs(OUTPUT_DIR_PROCESSED, exist_ok=True)
        os.makedirs(XML_OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(OUTPUT_FILE_ANNOTATION_TARGET_LIST), exist_ok=True)
        os.makedirs(os.path.dirname(OUTPUT_FILE_PREDICTION_LLM), exist_ok=True)
        os.makedirs(os.path.dirname(OUTPUT_FILE_FEATURES_FOR_EVALUATION), exist_ok=True)

        # Create dummy prompt files for LLM validation
        os.makedirs('prompts', exist_ok=True)
        with open('prompts/zero_shot_abstract.txt', 'w') as f: f.write('dummy')
        with open('prompts/zero_shot_fulltext.txt', 'w') as f: f.write('dummy')
        with open('prompts/few_shot_cot_fulltext.txt', 'w') as f: f.write('dummy')

        # Create dummy input files for pipelines that read them
        pd.DataFrame({'eid': ['d1'], 'doi': ['d1'], 'title': ['t1'], 'publication_year': ['2020'], 'citedby_count': [10]}).to_csv(OUTPUT_FILE_DATA_PAPERS, index=False)
        pd.DataFrame({'citing_paper_doi': ['c1'], 'fulltext_xml_path': ['path/to/xml'], 'download_status': ['success']}).to_csv(OUTPUT_FILE_CITING_PAPERS_WITH_PATHS, index=False)
        pd.DataFrame({'citing_paper_doi': ['c1'], 'is_data_used_gt': [1]}).to_csv(OUTPUT_FILE_ANNOTATION_TARGET_LIST, index=False)
        pd.DataFrame({'citing_paper_doi': ['c1'], 'abstract': ['abs'], 'full_text': ['full']}).to_csv(OUTPUT_FILE_SAMPLES_WITH_TEXT, index=False)
        pd.DataFrame({'citing_paper_doi': ['c1'], 'prediction_rule1': [1], 'prediction_rule2': [1]}).to_csv(OUTPUT_FILE_FEATURES_FOR_EVALUATION, index=False)
        pd.DataFrame({'citing_paper_doi': ['c1'], 'prediction_rule3_abstract': [1]}).to_csv(OUTPUT_FILE_PREDICTION_LLM, index=False)


    def tearDown(self):
        self._cleanup_files()

    def _cleanup_files(self):
        # Remove all files created by setup or pipelines
        files_to_remove = [
            OUTPUT_FILE_DATA_PAPERS,
            OUTPUT_FILE_CITING_PAPERS_WITH_PATHS,
            OUTPUT_FILE_ANNOTATION_TARGET_LIST,
            OUTPUT_FILE_SAMPLES_WITH_TEXT,
            OUTPUT_FILE_PREDICTION_LLM,
            OUTPUT_FILE_FEATURES_FOR_EVALUATION,
            os.path.join('prompts', 'zero_shot_abstract.txt'),
            os.path.join('prompts', 'zero_shot_fulltext.txt'),
            os.path.join('prompts', 'few_shot_cot_fulltext.txt'),
            os.path.join('results', 'tables', 'evaluation_metrics_summary.csv')
        ]
        for f in files_to_remove:
            if os.path.exists(f):
                os.remove(f)
        
        # Remove directories if empty
        dirs_to_remove = [
            OUTPUT_DIR_PROCESSED,
            XML_OUTPUT_DIR,
            os.path.dirname(OUTPUT_FILE_ANNOTATION_TARGET_LIST),
            os.path.dirname(OUTPUT_FILE_PREDICTION_LLM),
            os.path.dirname(OUTPUT_FILE_FEATURES_FOR_EVALUATION),
            'prompts',
            os.path.join('results', 'tables'),
            'results'
        ]
        for d in dirs_to_remove:
            if os.path.exists(d) and not os.listdir(d):
                os.rmdir(d)

    @patch('pipeline.main_pipeline.run_collect_data_pipeline')
    @patch('pipeline.main_pipeline.run_collect_citing_papers_pipeline')
    @patch('pipeline.main_pipeline.run_prepare_data_pipeline')
    @patch('pipeline.main_pipeline.run_llm_validation_pipeline')
    @patch('pipeline.main_pipeline.run_evaluate_results_pipeline')
    @patch('pipeline.main_pipeline.run_review_and_correct_pipeline')
    def test_main_pipeline_run_all(
        self, mock_review, mock_evaluate, mock_llm_validation,
        mock_prepare_data, mock_collect_citing, mock_collect_data
    ):
        """
        --run_all フラグが指定された場合に全てのパイプラインが実行されることをテスト。
        """
        args = argparse.Namespace(
            run_collect_data=False,
            run_collect_citing_papers=False,
            run_prepare_data=False,
            run_llm_validation=False,
            run_evaluate_results=False,
            run_review_and_correct=False,
            run_all=True,
            scopus_api_key="TEST_SCOPUS_KEY",
            gemini_api_key="TEST_GEMINI_KEY",
            gemini_model_name="test-model",
            random_state=42,
            min_citations=10,
            max_workers_download_xml=5,
            retry_failed_downloads=True,
            sample_size=100,
            run_abstract_prediction=True,
            run_fulltext_zeroshot_prediction=True,
            run_fulltext_fewshot_cot_prediction=False,
            retry_failed_abstract=True,
            retry_failed_fulltext_zeroshot=True,
            retry_failed_fulltext_fewshot_cot=False,
            llm_sleep_time=0.1,
            llm_timeout=60,
            best_model_column='prediction_rule3_gemini-2_5-flash'
        )

        main_pipeline(args)

        mock_collect_data.assert_called_once()
        mock_collect_citing.assert_called_once()
        mock_prepare_data.assert_called_once()
        mock_llm_validation.assert_called_once()
        mock_evaluate.assert_called_once()
        mock_review.assert_called_once()

        # Verify arguments passed to llm_validation_pipeline
        llm_validation_args = mock_llm_validation.call_args[1]
        self.assertTrue(llm_validation_args['run_abstract_prediction'])
        self.assertTrue(llm_validation_args['run_fulltext_zeroshot_prediction'])
        self.assertFalse(llm_validation_args['run_fulltext_fewshot_cot_prediction'])
        self.assertTrue(llm_validation_args['retry_failed_abstract'])
        self.assertTrue(llm_validation_args['retry_failed_fulltext_zeroshot'])
        self.assertFalse(llm_validation_args['retry_failed_fulltext_fewshot_cot'])

    @patch('pipeline.main_pipeline.run_collect_data_pipeline')
    @patch('pipeline.main_pipeline.run_collect_citing_papers_pipeline')
    @patch('pipeline.main_pipeline.run_prepare_data_pipeline')
    @patch('pipeline.main_pipeline.run_llm_validation_pipeline')
    @patch('pipeline.main_pipeline.run_evaluate_results_pipeline')
    @patch('pipeline.main_pipeline.run_review_and_correct_pipeline')
    def test_main_pipeline_selective_run(
        self, mock_review, mock_evaluate, mock_llm_validation,
        mock_prepare_data, mock_collect_citing, mock_collect_data
    ):
        """
        特定のパイプラインのみが実行されることをテスト。
        """
        args = argparse.Namespace(
            run_collect_data=True,
            run_collect_citing_papers=False,
            run_prepare_data=False,
            run_llm_validation=False,
            run_evaluate_results=False,
            run_review_and_correct=False,
            run_all=False,
            scopus_api_key="TEST_SCOPUS_KEY",
            gemini_api_key="TEST_GEMINI_KEY",
            gemini_model_name="test-model",
            random_state=42,
            min_citations=10,
            max_workers_download_xml=5,
            retry_failed_downloads=True,
            sample_size=100,
            run_abstract_prediction=False,
            run_fulltext_zeroshot_prediction=False,
            run_fulltext_fewshot_cot_prediction=False,
            retry_failed_abstract=False,
            retry_failed_fulltext_zeroshot=False,
            retry_failed_fulltext_fewshot_cot=False,
            llm_sleep_time=0.1,
            llm_timeout=60,
            best_model_column='prediction_rule3_gemini-2_5-flash'
        )

        main_pipeline(args)

        mock_collect_data.assert_called_once()
        mock_collect_citing.assert_not_called()
        mock_prepare_data.assert_not_called()
        mock_llm_validation.assert_not_called()
        mock_evaluate.assert_not_called()
        mock_review.assert_not_called()

if __name__ == '__main__':
    unittest.main()
