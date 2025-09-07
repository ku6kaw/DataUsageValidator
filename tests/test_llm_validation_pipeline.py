import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
from pipeline.llm_validation_pipeline import run_llm_validation_pipeline
from src.config import (
    GEMINI_API_KEY, GEMINI_MODEL_NAME,
    PROMPT_FILE_ZERO_SHOT_ABSTRACT, PROMPT_FILE_ZERO_SHOT_FULLTEXT, PROMPT_FILE_FEW_SHOT_COT_FULLTEXT,
    OUTPUT_FILE_SAMPLES_WITH_TEXT, OUTPUT_FILE_PREDICTION_LLM
)

class TestLlmValidationPipeline(unittest.TestCase):

    def setUp(self):
        self.test_samples_csv = "test_samples_with_text_pipeline.csv"
        self.test_predictions_csv = "test_prediction_llm_pipeline.csv"
        
        # Create dummy samples_with_text.csv
        self.dummy_samples_data = {
            'citing_paper_eid': ['eid1', 'eid2', 'eid3', 'eid4'],
            'citing_paper_doi': ['doi1', 'doi2', 'doi3', 'doi4'],
            'citing_paper_title': ['Citing Title 1', 'Citing Title 2', 'Citing Title 3', 'Citing Title 4'],
            'cited_data_paper_title': ['Cited Data 1', 'Cited Data 2', 'Cited Data 3', 'Cited Data 4'],
            'abstract': ['Abstract 1 content.', 'Abstract 2 content.', None, 'Abstract 4 content.'],
            'full_text': ['Full text 1 content.', None, 'Full text 3 content.', 'Full text 4 content.']
        }
        self.df_dummy_samples = pd.DataFrame(self.dummy_samples_data)
        self.df_dummy_samples.to_csv(self.test_samples_csv, index=False)

        # Create dummy prompt files
        for prompt_file in [PROMPT_FILE_ZERO_SHOT_ABSTRACT, PROMPT_FILE_ZERO_SHOT_FULLTEXT, PROMPT_FILE_FEW_SHOT_COT_FULLTEXT]:
            os.makedirs(os.path.dirname(prompt_file), exist_ok=True)
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(f"Dummy prompt for {os.path.basename(prompt_file)}")

    def tearDown(self):
        if os.path.exists(self.test_samples_csv):
            os.remove(self.test_samples_csv)
        if os.path.exists(self.test_predictions_csv):
            os.remove(self.test_predictions_csv)
        
        for prompt_file in [PROMPT_FILE_ZERO_SHOT_ABSTRACT, PROMPT_FILE_ZERO_SHOT_FULLTEXT, PROMPT_FILE_FEW_SHOT_COT_FULLTEXT]:
            if os.path.exists(prompt_file):
                os.remove(prompt_file)
            if os.path.exists(os.path.dirname(prompt_file)) and not os.listdir(os.path.dirname(prompt_file)):
                os.rmdir(os.path.dirname(prompt_file))

    @patch('pipeline.llm_validation_pipeline.configure_gemini_api')
    @patch('pipeline.llm_validation_pipeline.load_prompt_template')
    @patch('pipeline.llm_validation_pipeline.run_llm_prediction')
    @patch('pipeline.llm_validation_pipeline.save_llm_predictions')
    @patch('pipeline.llm_validation_pipeline.retry_llm_predictions')
    @patch('pandas.read_csv')
    def test_run_llm_validation_pipeline_all_predictions_and_retries(
        self, mock_read_csv, mock_retry, mock_save, mock_run_prediction, mock_load_prompt, mock_configure
    ):
        """
        LLM検証パイプラインが全ての設定で正常に実行される場合のテスト。
        """
        mock_read_csv.side_effect = [
            self.df_dummy_samples, # For initial df_samples load
            self.df_dummy_samples.dropna(subset=['abstract']), # For abstract targets
            self.df_dummy_samples.dropna(subset=['full_text']).drop_duplicates(subset=['citing_paper_doi']), # For fulltext zeroshot targets
            self.df_dummy_samples.dropna(subset=['full_text']).drop_duplicates(subset=['citing_paper_doi']), # For fulltext fewshot cot targets
            pd.DataFrame({ # For retry_llm_predictions (abstract)
                'citing_paper_doi': ['doi1', 'doi2'],
                'prediction_rule3_abstract': [0, -1]
            }),
            pd.DataFrame({ # For retry_llm_predictions (fulltext zeroshot)
                'citing_paper_doi': ['doi1', 'doi3'],
                'prediction_rule3_gemini-1_5-flash_zeroshot': [0, -1]
            }),
            pd.DataFrame({ # For retry_llm_predictions (fulltext fewshot cot)
                'citing_paper_doi': ['doi1', 'doi3'],
                'prediction_rule3_gemini-1_5-flash_few_shot_cot': [0, -1]
            }),
            self.df_dummy_samples.dropna(subset=['abstract']), # For retry_llm_predictions (abstract)
            self.df_dummy_samples.dropna(subset=['full_text']).drop_duplicates(subset=['citing_paper_doi']), # For retry_llm_predictions (fulltext zeroshot)
            self.df_dummy_samples.dropna(subset=['full_text']).drop_duplicates(subset=['citing_paper_doi']) # For retry_llm_predictions (fulltext fewshot cot)
        ]

        mock_load_prompt.return_value = "Dummy Prompt Template"
        mock_run_prediction.return_value = pd.DataFrame({
            'citing_paper_doi': ['doi1'],
            'prediction_test': [1]
        }) # Generic return for run_llm_prediction

        run_llm_validation_pipeline(
            api_key="TEST_KEY",
            model_name=GEMINI_MODEL_NAME,
            input_samples_csv=self.test_samples_csv,
            output_predictions_csv=self.test_predictions_csv,
            run_abstract_prediction=True,
            run_fulltext_zeroshot_prediction=True,
            run_fulltext_fewshot_cot_prediction=True,
            retry_failed_abstract=True,
            retry_failed_fulltext_zeroshot=True,
            retry_failed_fulltext_fewshot_cot=True,
            sleep_time=0.01, # Reduce sleep time for tests
            timeout=10
        )

        mock_configure.assert_called_once_with(api_key="TEST_KEY")
        self.assertEqual(mock_load_prompt.call_count, 3) # Abstract, Zero-shot, Few-shot CoT
        self.assertEqual(mock_run_prediction.call_count, 3) # Abstract, Zero-shot, Few-shot CoT
        self.assertEqual(mock_save.call_count, 3) # Abstract, Zero-shot, Few-shot CoT
        self.assertEqual(mock_retry.call_count, 3) # Abstract, Zero-shot, Few-shot CoT

    @patch('pipeline.llm_validation_pipeline.configure_gemini_api')
    @patch('pipeline.llm_validation_pipeline.load_prompt_template')
    @patch('pipeline.llm_validation_pipeline.run_llm_prediction')
    @patch('pipeline.llm_validation_pipeline.save_llm_predictions')
    @patch('pipeline.llm_validation_pipeline.retry_llm_predictions')
    @patch('pandas.read_csv')
    def test_run_llm_validation_pipeline_no_samples(
        self, mock_read_csv, mock_retry, mock_save, mock_run_prediction, mock_load_prompt, mock_configure
    ):
        """
        入力サンプルデータがない場合、パイプラインがスキップされるテスト。
        """
        mock_read_csv.return_value = pd.DataFrame() # 空のDataFrameを返す

        run_llm_validation_pipeline(
            input_samples_csv=self.test_samples_csv,
            run_abstract_prediction=True,
            run_fulltext_zeroshot_prediction=True
        )

        mock_configure.assert_called_once()
        mock_load_prompt.assert_not_called()
        mock_run_prediction.assert_not_called()
        mock_save.assert_not_called()
        mock_retry.assert_not_called()

    @patch('pipeline.llm_validation_pipeline.configure_gemini_api')
    @patch('pipeline.llm_validation_pipeline.load_prompt_template')
    @patch('pipeline.llm_validation_pipeline.run_llm_prediction')
    @patch('pipeline.llm_validation_pipeline.save_llm_predictions')
    @patch('pipeline.llm_validation_pipeline.retry_llm_predictions')
    @patch('pandas.read_csv')
    def test_run_llm_validation_pipeline_abstract_only(
        self, mock_read_csv, mock_retry, mock_save, mock_run_prediction, mock_load_prompt, mock_configure
    ):
        """
        アブストラクト予測のみ実行される場合のテスト。
        """
        mock_read_csv.side_effect = [
            self.df_dummy_samples, # For initial df_samples load
            self.df_dummy_samples.dropna(subset=['abstract']) # For abstract targets
        ]
        mock_load_prompt.return_value = "Dummy Abstract Prompt"
        mock_run_prediction.return_value = pd.DataFrame({
            'citing_paper_doi': ['doi1', 'doi2', 'doi4'],
            'prediction_rule3_abstract': [1, 0, 1]
        })

        run_llm_validation_pipeline(
            input_samples_csv=self.test_samples_csv,
            run_abstract_prediction=True,
            run_fulltext_zeroshot_prediction=False,
            run_fulltext_fewshot_cot_prediction=False,
            retry_failed_abstract=False,
            retry_failed_fulltext_zeroshot=False,
            retry_failed_fulltext_fewshot_cot=False
        )

        mock_configure.assert_called_once()
        mock_load_prompt.assert_called_once_with(PROMPT_FILE_ZERO_SHOT_ABSTRACT)
        mock_run_prediction.assert_called_once()
        mock_save.assert_called_once()
        mock_retry.assert_not_called()

    @patch('pipeline.llm_validation_pipeline.configure_gemini_api')
    @patch('pipeline.llm_validation_pipeline.load_prompt_template')
    @patch('pipeline.llm_validation_pipeline.run_llm_prediction')
    @patch('pipeline.llm_validation_pipeline.save_llm_predictions')
    @patch('pipeline.llm_validation_pipeline.retry_llm_predictions')
    @patch('pandas.read_csv')
    def test_run_llm_validation_pipeline_no_abstract_available(
        self, mock_read_csv, mock_retry, mock_save, mock_run_prediction, mock_load_prompt, mock_configure
    ):
        """
        アブストラクトが利用可能な論文がない場合、アブストラクト予測がスキップされるテスト。
        """
        df_no_abstract = self.df_dummy_samples.copy()
        df_no_abstract['abstract'] = None
        mock_read_csv.side_effect = [
            df_no_abstract, # For initial df_samples load
            df_no_abstract.dropna(subset=['abstract']) # For abstract targets (will be empty)
        ]

        run_llm_validation_pipeline(
            input_samples_csv=self.test_samples_csv,
            run_abstract_prediction=True,
            run_fulltext_zeroshot_prediction=False,
            run_fulltext_fewshot_cot_prediction=False
        )

        mock_configure.assert_called_once()
        mock_load_prompt.assert_called_once_with(PROMPT_FILE_ZERO_SHOT_ABSTRACT)
        mock_run_prediction.assert_not_called() # Should be skipped
        mock_save.assert_not_called()
        mock_retry.assert_not_called()

if __name__ == '__main__':
    unittest.main()
