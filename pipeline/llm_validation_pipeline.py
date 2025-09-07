import pandas as pd
import os
from src.llm_validator import (
    configure_gemini_api,
    load_prompt_template,
    run_llm_prediction,
    save_llm_predictions,
    retry_llm_predictions
)
from src.config import (
    GEMINI_API_KEY, GEMINI_MODEL_NAME,
    PROMPT_FILE_ZERO_SHOT_ABSTRACT, PROMPT_FILE_ZERO_SHOT_FULLTEXT, PROMPT_FILE_FEW_SHOT_COT_FULLTEXT,
    OUTPUT_FILE_SAMPLES_WITH_TEXT, OUTPUT_FILE_PREDICTION_LLM
)

def run_llm_validation_pipeline(
    api_key: str = GEMINI_API_KEY,
    model_name: str = GEMINI_MODEL_NAME,
    input_samples_csv: str = OUTPUT_FILE_SAMPLES_WITH_TEXT,
    output_predictions_csv: str = OUTPUT_FILE_PREDICTION_LLM,
    run_abstract_prediction: bool = True,
    run_fulltext_zeroshot_prediction: bool = True,
    run_fulltext_fewshot_cot_prediction: bool = False,
    retry_failed_abstract: bool = True,
    retry_failed_fulltext_zeroshot: bool = True,
    retry_failed_fulltext_fewshot_cot: bool = False,
    sleep_time: float = 1.0,
    timeout: int = 180
):
    """
    LLM検証パイプラインを実行します。
    アブストラクトおよび全文を用いた予測、および失敗した予測の再試行が含まれます。

    Args:
        api_key (str): Gemini APIキー。
        model_name (str): 使用するGeminiモデルの名前。
        input_samples_csv (str): テキスト抽出済み論文のCSVパス。
        output_predictions_csv (str): LLM予測結果を保存するCSVパス。
        run_abstract_prediction (bool): アブストラクトを用いた予測を実行するかどうか。
        run_fulltext_zeroshot_prediction (bool): 全文を用いたZero-shot予測を実行するかどうか。
        run_fulltext_fewshot_cot_prediction (bool): 全文を用いたFew-shot CoT予測を実行するかどうか。
        retry_failed_abstract (bool): 失敗したアブストラクト予測を再試行するかどうか。
        retry_failed_fulltext_zeroshot (bool): 失敗した全文Zero-shot予測を再試行するかどうか。
        retry_failed_fulltext_fewshot_cot (bool): 失敗した全文Few-shot CoT予測を再試行するかどうか。
        sleep_time (float): APIリクエスト間の待機時間（秒）。
        timeout (int): APIリクエストのタイムアウト時間（秒）。
    """
    print("--- LLM検証フェーズ開始 ---")

    try:
        configure_gemini_api(api_key=api_key)
        df_samples = pd.read_csv(input_samples_csv)
        if df_samples.empty:
            print("処理対象のサンプルデータがないため、LLM検証をスキップします。")
            print("--- LLM検証フェーズ完了 ---")
            return

        # アブストラクト予測
        if run_abstract_prediction:
            print("\n--- アブストラクトを用いたLLM予測を開始 ---")
            prompt_template_abstract = load_prompt_template(PROMPT_FILE_ZERO_SHOT_ABSTRACT)
            df_abstract_targets = df_samples.dropna(subset=['abstract']).copy()
            if not df_abstract_targets.empty:
                df_predictions_abstract = run_llm_prediction(
                    df_to_process=df_abstract_targets,
                    prompt_template=prompt_template_abstract,
                    text_column='abstract',
                    prediction_column_name='prediction_rule3_abstract',
                    model_name=model_name,
                    api_key=api_key,
                    sleep_time=sleep_time,
                    timeout=timeout
                )
                save_llm_predictions(df_predictions_abstract, output_file_path=output_predictions_csv, prediction_column_name='prediction_rule3_abstract')
            else:
                print("アブストラクトが利用可能な論文がないため、アブストラクト予測をスキップします。")

        # 全文Zero-shot予測
        if run_fulltext_zeroshot_prediction:
            print("\n--- 全文を用いたZero-shot LLM予測を開始 ---")
            prompt_template_fulltext_zeroshot = load_prompt_template(PROMPT_FILE_ZERO_SHOT_FULLTEXT)
            df_fulltext_targets = df_samples.dropna(subset=['full_text']).copy()
            df_fulltext_targets.drop_duplicates(subset=['citing_paper_doi'], inplace=True, keep='first')
            df_fulltext_targets.reset_index(drop=True, inplace=True)
            if not df_fulltext_targets.empty:
                prediction_column_name_zeroshot = f"prediction_rule3_{model_name.replace('.', '_')}_zeroshot"
                df_predictions_fulltext_zeroshot = run_llm_prediction(
                    df_to_process=df_fulltext_targets,
                    prompt_template=prompt_template_fulltext_zeroshot,
                    text_column='full_text',
                    prediction_column_name=prediction_column_name_zeroshot,
                    model_name=model_name,
                    api_key=api_key,
                    sleep_time=sleep_time,
                    timeout=timeout
                )
                save_llm_predictions(df_predictions_fulltext_zeroshot, output_file_path=output_predictions_csv, prediction_column_name=prediction_column_name_zeroshot)
            else:
                print("全文が利用可能な論文がないため、全文Zero-shot予測をスキップします。")

        # 全文Few-shot CoT予測
        if run_fulltext_fewshot_cot_prediction:
            print("\n--- 全文を用いたFew-shot CoT LLM予測を開始 ---")
            prompt_template_fulltext_fewshot_cot = load_prompt_template(PROMPT_FILE_FEW_SHOT_COT_FULLTEXT)
            df_fulltext_targets = df_samples.dropna(subset=['full_text']).copy()
            df_fulltext_targets.drop_duplicates(subset=['citing_paper_doi'], inplace=True, keep='first')
            df_fulltext_targets.reset_index(drop=True, inplace=True)
            if not df_fulltext_targets.empty:
                prediction_column_name_fewshot_cot = f"prediction_rule3_{model_name.replace('.', '_')}_few_shot_cot"
                df_predictions_fulltext_fewshot_cot = run_llm_prediction(
                    df_to_process=df_fulltext_targets,
                    prompt_template=prompt_template_fulltext_fewshot_cot,
                    text_column='full_text',
                    prediction_column_name=prediction_column_name_fewshot_cot,
                    model_name=model_name,
                    api_key=api_key,
                    sleep_time=sleep_time,
                    timeout=timeout
                )
                save_llm_predictions(df_predictions_fulltext_fewshot_cot, output_file_path=output_predictions_csv, prediction_column_name=prediction_column_name_fewshot_cot)
            else:
                print("全文が利用可能な論文がないため、全文Few-shot CoT予測をスキップします。")

        # 失敗した予測の再試行
        if retry_failed_abstract and run_abstract_prediction:
            print("\n--- 失敗したアブストラクト予測の再試行を開始 ---")
            retry_llm_predictions(
                input_samples_csv=input_samples_csv,
                input_predictions_csv=output_predictions_csv,
                column_to_retry='prediction_rule3_abstract',
                prompt_file_path=PROMPT_FILE_ZERO_SHOT_ABSTRACT,
                text_column='abstract',
                model_name=model_name,
                api_key=api_key,
                sleep_time=sleep_time,
                timeout=timeout
            )
        
        if retry_failed_fulltext_zeroshot and run_fulltext_zeroshot_prediction:
            print("\n--- 失敗した全文Zero-shot予測の再試行を開始 ---")
            retry_llm_predictions(
                input_samples_csv=input_samples_csv,
                input_predictions_csv=output_predictions_csv,
                column_to_retry=f"prediction_rule3_{model_name.replace('.', '_')}_zeroshot",
                prompt_file_path=PROMPT_FILE_ZERO_SHOT_FULLTEXT,
                text_column='full_text',
                model_name=model_name,
                api_key=api_key,
                sleep_time=sleep_time,
                timeout=timeout
            )

        if retry_failed_fulltext_fewshot_cot and run_fulltext_fewshot_cot_prediction:
            print("\n--- 失敗した全文Few-shot CoT予測の再試行を開始 ---")
            retry_llm_predictions(
                input_samples_csv=input_samples_csv,
                input_predictions_csv=output_predictions_csv,
                column_to_retry=f"prediction_rule3_{model_name.replace('.', '_')}_few_shot_cot",
                prompt_file_path=PROMPT_FILE_FEW_SHOT_COT_FULLTEXT,
                text_column='full_text',
                model_name=model_name,
                api_key=api_key,
                sleep_time=sleep_time,
                timeout=timeout
            )

    except FileNotFoundError as e:
        print(f"LLM検証パイプラインに必要なファイルが見つかりません: {e}")
    except Exception as e:
        print(f"LLM検証パイプラインの実行中にエラーが発生しました: {e}")
    
    print("--- LLM検証フェーズ完了 ---")

if __name__ == "__main__":
    # このスクリプトを直接実行する場合の例
    run_llm_validation_pipeline(
        run_abstract_prediction=True,
        run_fulltext_zeroshot_prediction=True,
        run_fulltext_fewshot_cot_prediction=False, # 必要に応じてTrueに変更
        retry_failed_abstract=True,
        retry_failed_fulltext_zeroshot=True,
        retry_failed_fulltext_fewshot_cot=False
    )
