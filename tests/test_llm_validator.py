import unittest
import pandas as pd
import os
from unittest.mock import patch, MagicMock
import google.generativeai as genai
import requests

from src.llm_validator import (
    configure_gemini_api,
    load_prompt_template,
    run_llm_prediction,
    save_llm_predictions,
    retry_llm_predictions,
    _call_gemini_api
)
from src.config import (
    GEMINI_API_KEY, GEMINI_MODEL_NAME,
    PROMPT_FILE_ZERO_SHOT_ABSTRACT,
    OUTPUT_FILE_SAMPLES_WITH_TEXT, OUTPUT_FILE_PREDICTION_LLM,
    OUTPUT_DIR_PROCESSED
)

class TestLlmValidator(unittest.TestCase):

    def setUp(self):
        self.test_samples_csv = "test_samples_with_text.csv"
        self.test_predictions_csv = "test_prediction_llm.csv"
        self.test_prompt_file = "test_prompt.txt"
        self.test_output_dir = "test_llm_output"
        os.makedirs(self.test_output_dir, exist_ok=True)

        # ダミーのsamples_with_text.csv
        self.dummy_samples_data = {
            'citing_paper_eid': ['eid1', 'eid2', 'eid3'],
            'citing_paper_doi': ['doi1', 'doi2', 'doi3'],
            'citing_paper_title': ['Citing Title 1', 'Citing Title 2', 'Citing Title 3'],
            'cited_data_paper_title': ['Cited Data 1', 'Cited Data 2', 'Cited Data 3'],
            'abstract': ['Abstract 1 content.', 'Abstract 2 content.', 'Abstract 3 content.'],
            'full_text': ['Full text 1 content.', 'Full text 2 content.', 'Full text 3 content.']
        }
        self.df_dummy_samples = pd.DataFrame(self.dummy_samples_data)
        self.df_dummy_samples.to_csv(self.test_samples_csv, index=False)

        # ダミーのprompt.txt
        with open(self.test_prompt_file, "w", encoding="utf-8") as f:
            f.write("Cited: {cited_data_paper_title}\nCiting: {citing_paper_title}\nText: {citing_paper_text}")

    def tearDown(self):
        if os.path.exists(self.test_samples_csv):
            os.remove(self.test_samples_csv)
        if os.path.exists(self.test_predictions_csv):
            os.remove(self.test_predictions_csv)
        if os.path.exists(self.test_prompt_file):
            os.remove(self.test_prompt_file)
        if os.path.exists(self.test_output_dir):
            os.rmdir(self.test_output_dir)

    @patch('google.generativeai.configure')
    def test_configure_gemini_api(self, mock_configure):
        configure_gemini_api(api_key="TEST_KEY")
        mock_configure.assert_called_once_with(api_key="TEST_KEY")

    def test_load_prompt_template_success(self):
        template = load_prompt_template(self.test_prompt_file)
        self.assertIn("Cited:", template)

    def test_load_prompt_template_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_prompt_template("non_existent_prompt.txt")

    @patch('requests.post')
    def test_call_gemini_api_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'candidates': [{'content': {'parts': [{'text': '{"decision": "Used"}'}]}}]
        }
        mock_post.return_value = mock_response

        response_text = _call_gemini_api("test prompt", "test-model", "test-key")
        self.assertEqual(response_text, '{"decision": "Used"}')
        mock_post.assert_called_once()

    @patch('src.llm_validator._call_gemini_api', return_value='{"decision": "Used"}')
    def test_run_llm_prediction_success(self, mock_call_api):
        df_predictions = run_llm_prediction(
            df_to_process=self.df_dummy_samples,
            prompt_template="Test: {citing_paper_text}",
            text_column='abstract',
            prediction_column_name='prediction_test',
            api_key="TEST_KEY"
        )
        self.assertFalse(df_predictions.empty)
        self.assertEqual(len(df_predictions), 3)
        self.assertIn('prediction_test', df_predictions.columns)
        self.assertTrue(all(df_predictions['prediction_test'] == 1))
        self.assertEqual(mock_call_api.call_count, 3)

    @patch('src.llm_validator._call_gemini_api', side_effect=[
        '{"decision": "Used"}',
        Exception("API Error"), # Simulate an error for one row
        '{"decision": "Not Used"}'
    ])
    def test_run_llm_prediction_with_error(self, mock_call_api):
        df_predictions = run_llm_prediction(
            df_to_process=self.df_dummy_samples,
            prompt_template="Test: {citing_paper_text}",
            text_column='abstract',
            prediction_column_name='prediction_test',
            api_key="TEST_KEY"
        )
        self.assertFalse(df_predictions.empty)
        self.assertEqual(len(df_predictions), 3)
        self.assertIn('prediction_test', df_predictions.columns)
        self.assertEqual(df_predictions.iloc[0]['prediction_test'], 1)
        self.assertEqual(df_predictions.iloc[1]['prediction_test'], -1) # Expect -1 for error
        self.assertEqual(df_predictions.iloc[2]['prediction_test'], 0)
        self.assertEqual(mock_call_api.call_count, 3)

    def test_save_llm_predictions_new_file(self):
        df_to_save = pd.DataFrame({
            'citing_paper_eid': ['eid1'],
            'citing_paper_doi': ['doi1'],
            'citing_paper_title': ['Title 1'],
            'cited_data_paper_title': ['Data Title A'],
            'prediction_test': [1]
        })
        save_llm_predictions(df_to_save, output_file_path=self.test_predictions_csv, prediction_column_name='prediction_test')
        self.assertTrue(os.path.exists(self.test_predictions_csv))
        loaded_df = pd.read_csv(self.test_predictions_csv)
        self.assertEqual(len(loaded_df), 1)
        self.assertEqual(loaded_df.iloc[0]['prediction_test'], 1)

    def test_save_llm_predictions_update_existing_file(self):
        # 既存のファイルを作成
        existing_data = {
            'citing_paper_eid': ['eid1', 'eid2'],
            'citing_paper_doi': ['doi1', 'doi2'],
            'citing_paper_title': ['Title 1', 'Title 2'],
            'cited_data_paper_title': ['Data Title A', 'Data Title B'],
            'prediction_rule3_abstract': [0, 1]
        }
        pd.DataFrame(existing_data).to_csv(self.test_predictions_csv, index=False)

        # 更新データ
        update_data = {
            'citing_paper_doi': ['doi1', 'doi3'],
            'prediction_rule3_fulltext': [1, 0]
        }
        df_update = pd.DataFrame(update_data)

        save_llm_predictions(df_update, output_file_path=self.test_predictions_csv, prediction_column_name='prediction_rule3_fulltext')
        loaded_df = pd.read_csv(self.test_predictions_csv)
        self.assertEqual(len(loaded_df), 2) # doi3はmasterにないのでマージされない
        self.assertIn('prediction_rule3_fulltext', loaded_df.columns)
        self.assertEqual(loaded_df.loc[loaded_df['citing_paper_doi'] == 'doi1', 'prediction_rule3_fulltext'].iloc[0], 1)
        self.assertTrue(pd.isna(loaded_df.loc[loaded_df['citing_paper_doi'] == 'doi2', 'prediction_rule3_fulltext'].iloc[0]))

    @patch('src.llm_validator.run_llm_prediction', return_value=pd.DataFrame({
        'citing_paper_doi': ['doi2'],
        'prediction_test_retry': [1]
    }))
    @patch('src.llm_validator.load_prompt_template', return_value="Test Prompt")
    @patch('src.llm_validator.configure_gemini_api', return_value=None)
    def test_retry_llm_predictions(self, mock_configure, mock_load_prompt, mock_run_prediction):
        # ダミーのsamples_with_text.csv
        dummy_samples = {
            'citing_paper_eid': ['eid1', 'eid2'],
            'citing_paper_doi': ['doi1', 'doi2'],
            'citing_paper_title': ['Title 1', 'Title 2'],
            'cited_data_paper_title': ['Data A', 'Data B'],
            'abstract': ['Abs 1', 'Abs 2']
        }
        pd.DataFrame(dummy_samples).to_csv(self.test_samples_csv, index=False)

        # ダミーのprediction_llm.csv (エラーを含む)
        dummy_predictions = {
            'citing_paper_eid': ['eid1', 'eid2'],
            'citing_paper_doi': ['doi1', 'doi2'],
            'citing_paper_title': ['Title 1', 'Title 2'],
            'cited_data_paper_title': ['Data A', 'Data B'],
            'prediction_test_retry': [0, -1] # doi2がエラー
        }
        pd.DataFrame(dummy_predictions).to_csv(self.test_predictions_csv, index=False)

        retry_llm_predictions(
            input_samples_csv=self.test_samples_csv,
            input_predictions_csv=self.test_predictions_csv,
            column_to_retry='prediction_test_retry',
            prompt_file_path=self.test_prompt_file,
            text_column='abstract',
            api_key="TEST_KEY"
        )

        mock_run_prediction.assert_called_once()
        # run_llm_predictionに渡されたDataFrameがdoi2のみであることを確認
        called_df = mock_run_prediction.call_args[0][0]
        self.assertEqual(len(called_df), 1)
        self.assertEqual(called_df.iloc[0]['citing_paper_doi'], 'doi2')

        # 結果が更新されたか確認
        loaded_df = pd.read_csv(self.test_predictions_csv)
        self.assertEqual(loaded_df.loc[loaded_df['citing_paper_doi'] == 'doi2', 'prediction_test_retry'].iloc[0], 1)

if __name__ == '__main__':
    unittest.main()
