import pandas as pd
import os
import google.generativeai as genai
import json
import time
from tqdm import tqdm
import re
import requests # For direct API calls in fulltext prediction

from src.config import (
    GEMINI_API_KEY, GEMINI_MODEL_NAME,
    PROMPT_FILE_ZERO_SHOT_ABSTRACT, PROMPT_FILE_ZERO_SHOT_FULLTEXT, PROMPT_FILE_FEW_SHOT_COT_FULLTEXT,
    OUTPUT_FILE_SAMPLES_WITH_TEXT, OUTPUT_FILE_PREDICTION_LLM,
    OUTPUT_DIR_PROCESSED
)

def configure_gemini_api(api_key: str = GEMINI_API_KEY):
    """
    Gemini APIキーを設定します。
    """
    try:
        genai.configure(api_key=api_key)
        print("Gemini APIキーの設定が完了しました。")
    except Exception as e:
        print(f"Gemini APIキーの設定でエラー: {e}")
        raise

def load_prompt_template(prompt_file_path: str) -> str:
    """
    指定されたパスからプロンプトテンプレートを読み込みます。

    Args:
        prompt_file_path (str): プロンプトファイルのパス。

    Returns:
        str: 読み込まれたプロンプトテンプレート。
    """
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        print(f"プロンプトファイル '{prompt_file_path}' を正常に読み込みました。")
        return prompt_template
    except FileNotFoundError:
        print(f"エラー: プロンプトファイル '{prompt_file_path}' が見つかりません。")
        raise
    except Exception as e:
        print(f"プロンプトファイルの読み込み中にエラーが発生しました: {e}")
        raise

def _call_gemini_api(prompt: str, model_name: str, api_key: str, timeout: int = 120) -> str or None:
    """
    Gemini APIを直接呼び出し、レスポンステキストを返します。
    """
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(api_url, json=payload, timeout=timeout)
        response.raise_for_status()
        response_json = response.json()
        return response_json['candidates'][0]['content']['parts'][0]['text']
    except requests.exceptions.RequestException as e:
        print(f"APIリクエストエラー: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"APIレスポンス解析エラー: {e}")
        return None
    except Exception as e:
        print(f"予期せぬエラー: {e}")
        return None

def run_llm_prediction(
    df_to_process: pd.DataFrame,
    prompt_template: str,
    text_column: str,
    prediction_column_name: str,
    model_name: str = GEMINI_MODEL_NAME,
    api_key: str = GEMINI_API_KEY,
    sleep_time: float = 1.0,
    timeout: int = 120
) -> pd.DataFrame:
    """
    LLMを使用して論文のデータ利用を予測します。

    Args:
        df_to_process (pd.DataFrame): 処理対象の論文データDataFrame。
        prompt_template (str): LLMに渡すプロンプトのテンプレート。
        text_column (str): プロンプトに埋め込むテキストが含まれるDataFrameの列名（'abstract'または'full_text'）。
        prediction_column_name (str): 予測結果を格納する新しい列名。
        model_name (str): 使用するGeminiモデルの名前。
        api_key (str): Gemini APIキー。
        sleep_time (float): APIリクエスト間の待機時間（秒）。
        timeout (int): APIリクエストのタイムアウト時間（秒）。

    Returns:
        pd.DataFrame: 予測結果が追加されたDataFrame。
    """
    if df_to_process.empty:
        print("処理対象のデータがありません。LLM予測をスキップします。")
        return pd.DataFrame()

    predictions = []

    print(f"合計 {len(df_to_process)} 件の論文に対してLLM({text_column}, {model_name})の判定を実行します。")

    for index, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc=f"LLM({text_column})で判定中"):
        
        text_content = row.get(text_column, '')
        # フルテキストの場合、トークン数上限を考慮して切り詰める
        if text_column == 'full_text':
            text_content = text_content[:30000] # 仮のトークン上限
        
        prompt = prompt_template.format(
            cited_data_paper_title=row['cited_data_paper_title'],
            citing_paper_title=row['citing_paper_title'],
            citing_paper_text=text_content
        )
        
        prediction = -1 # デフォルトはエラー値
        try:
            response_text = _call_gemini_api(prompt, model_name, api_key, timeout)
            
            if response_text:
                match = re.search(r"\{.*\}", response_text, re.DOTALL)
                json_text = match.group(0) if match else response_text
                json_response = json.loads(json_text)
                
                decision = json_response.get('decision')
                prediction = 1 if decision == "Used" else 0
            
        except Exception as e:
            print(f"警告: DOI {row['citing_paper_doi']} の処理中にエラーが発生しました: {e}")
            
        predictions.append(prediction)
        time.sleep(sleep_time)

    df_results = df_to_process.copy()
    df_results[prediction_column_name] = predictions
    return df_results

def save_llm_predictions(df: pd.DataFrame, output_file_path: str = OUTPUT_FILE_PREDICTION_LLM, prediction_column_name: str = None):
    """
    LLMの予測結果をCSVファイルに保存します。

    Args:
        df (pd.DataFrame): 予測結果を含むDataFrame。
        output_file_path (str): 保存先のCSVファイルパス。
        prediction_column_name (str): 予測結果の列名（サマリー表示用）。
    """
    if not df.empty:
        # output_file_pathがファイル名のみの場合、カレントディレクトリを対象とする
        output_dir = os.path.dirname(output_file_path)
        if not output_dir:
            output_dir = '.' # Current directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 既存のファイルがあれば読み込み、マージする
        try:
            existing_df = pd.read_csv(output_file_path)
            # 既存のデータと新しい予測結果をマージ（結合）
            # 既に同名カラムがあれば上書きし、なければ追加する
            if prediction_column_name and prediction_column_name in existing_df.columns:
                existing_df.drop(columns=[prediction_column_name], inplace=True)
            
            # citing_paper_doiをキーとしてマージ
            df_to_save = pd.merge(existing_df, df[['citing_paper_doi', prediction_column_name]], on='citing_paper_doi', how='left')
        except FileNotFoundError:
            # ファイルが存在しない場合は、新しいDataFrameをそのまま保存
            df_to_save = df[['citing_paper_eid', 'citing_paper_doi', 'citing_paper_title', 'cited_data_paper_title', prediction_column_name]].copy()
        
        df_to_save.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print(f"\n処理完了。LLMの予測結果を '{output_file_path}' に保存しました。")
        if prediction_column_name:
            print(f"\n--- 保存された結果の内訳 ({prediction_column_name}) ---")
            print(df_to_save[prediction_column_name].value_counts())
    else:
        print("保存するデータがありませんでした。")

def retry_llm_predictions(
    input_samples_csv: str = OUTPUT_FILE_SAMPLES_WITH_TEXT,
    input_predictions_csv: str = OUTPUT_FILE_PREDICTION_LLM,
    column_to_retry: str = 'prediction_rule3_abstract',
    prompt_file_path: str = PROMPT_FILE_ZERO_SHOT_ABSTRACT,
    text_column: str = 'abstract',
    model_name: str = GEMINI_MODEL_NAME,
    api_key: str = GEMINI_API_KEY,
    sleep_time: float = 1.5,
    timeout: int = 180
):
    """
    LLM予測で失敗した（-1）論文の予測を再試行します。

    Args:
        input_samples_csv (str): テキスト抽出済み論文のCSVパス。
        input_predictions_csv (str): 既存のLLM予測結果CSVパス。
        column_to_retry (str): 再試行する予測結果の列名。
        prompt_file_path (str): 再試行に使用するプロンプトファイルのパス。
        text_column (str): プロンプトに埋め込むテキストが含まれるDataFrameの列名。
        model_name (str): 使用するGeminiモデルの名前。
        api_key (str): Gemini APIキー。
        sleep_time (float): APIリクエスト間の待機時間（秒）。
        timeout (int): APIリクエストのタイムアウト時間（秒）。
    """
    try:
        df_samples = pd.read_csv(input_samples_csv)
        df_predictions = pd.read_csv(input_predictions_csv)

        if column_to_retry not in df_predictions.columns:
            print(f"エラー: '{column_to_retry}' カラムが予測結果ファイルに見つかりません。再試行をスキップします。")
            return

        retry_dois = df_predictions[df_predictions[column_to_retry] == -1]['citing_paper_doi']
        df_to_retry = df_samples[df_samples['citing_paper_doi'].isin(retry_dois)].copy()
        
        if df_to_retry.empty:
            print("エラー(-1)が記録された論文はありませんでした。再試行は不要です。")
            return

        print(f"エラー(-1)が記録された {len(df_to_retry)} 件を特定しました。再試行を開始します。")
        
        prompt_template = load_prompt_template(prompt_file_path)

        retry_results_df = run_llm_prediction(
            df_to_process=df_to_retry,
            prompt_template=prompt_template,
            text_column=text_column,
            prediction_column_name=column_to_retry,
            model_name=model_name,
            api_key=api_key,
            sleep_time=sleep_time,
            timeout=timeout
        )

        # 元の予測結果を更新
        df_predictions.set_index('citing_paper_doi', inplace=True)
        retry_results_df.set_index('citing_paper_doi', inplace=True)
        df_predictions.update(retry_results_df[[column_to_retry]])
        df_predictions.reset_index(inplace=True)
        
        save_llm_predictions(df_predictions, output_file_path=input_predictions_csv, prediction_column_name=column_to_retry)
        print(f"\n--- 最新の予測結果の内訳（{column_to_retry}） ---")
        print(df_predictions[column_to_retry].value_counts())

    except FileNotFoundError as e:
        print(f"必要なファイルが見つかりません: {e}")
    except Exception as e:
        print(f"LLM予測の再試行中にエラーが発生しました: {e}")

def main_llm_abstract_prediction():
    """
    アブストラクトを用いたLLM予測のメイン処理を実行します。
    """
    try:
        configure_gemini_api()
        prompt_template = load_prompt_template(PROMPT_FILE_ZERO_SHOT_ABSTRACT)
        
        df_to_process = pd.read_csv(OUTPUT_FILE_SAMPLES_WITH_TEXT)
        df_to_process.dropna(subset=['abstract'], inplace=True) # アブストラクトが空の行は除外
        
        df_predictions = run_llm_prediction(
            df_to_process=df_to_process,
            prompt_template=prompt_template,
            text_column='abstract',
            prediction_column_name='prediction_rule3_abstract'
        )
        
        save_llm_predictions(df_predictions, output_file_path=OUTPUT_FILE_PREDICTION_LLM, prediction_column_name='prediction_rule3_abstract')

    except Exception as e:
        print(f"LLMアブストラクト予測のメイン処理中にエラーが発生しました: {e}")

def main_llm_fulltext_prediction(
    prompt_file_path: str = PROMPT_FILE_ZERO_SHOT_FULLTEXT,
    prediction_column_suffix: str = 'zeroshot',
    model_name: str = GEMINI_MODEL_NAME,
    sleep_time: float = 1.5,
    timeout: int = 180
):
    """
    全文を用いたLLM予測のメイン処理を実行します。
    """
    try:
        configure_gemini_api()
        prompt_template = load_prompt_template(prompt_file_path)
        
        df_to_process = pd.read_csv(OUTPUT_FILE_SAMPLES_WITH_TEXT)
        df_to_process.dropna(subset=['full_text'], inplace=True) # フルテキストが空の行は除外
        df_to_process.drop_duplicates(subset=['citing_paper_doi'], inplace=True, keep='first')
        df_to_process.reset_index(drop=True, inplace=True)
        
        prediction_column_name = f"prediction_rule3_{model_name.replace('.', '_')}_{prediction_column_suffix}"

        df_predictions = run_llm_prediction(
            df_to_process=df_to_process,
            prompt_template=prompt_template,
            text_column='full_text',
            prediction_column_name=prediction_column_name,
            model_name=model_name,
            sleep_time=sleep_time,
            timeout=timeout
        )
        
        save_llm_predictions(df_predictions, output_file_path=OUTPUT_FILE_PREDICTION_LLM, prediction_column_name=prediction_column_name)

    except Exception as e:
        print(f"LLM全文予測のメイン処理中にエラーが発生しました: {e}")

if __name__ == "__main__":
    # 例としてアブストラクト予測を実行
    main_llm_abstract_prediction()
    # フルテキスト予測（Zero-shot）の例
    # main_llm_fulltext_prediction(
    #     prompt_file_path=PROMPT_FILE_ZERO_SHOT_FULLTEXT,
    #     prediction_column_suffix='zeroshot',
    #     model_name=GEMINI_MODEL_NAME
    # )
    # フルテキスト予測（Few-shot CoT）の例
    # main_llm_fulltext_prediction(
    #     prompt_file_path=PROMPT_FILE_FEW_SHOT_COT_FULLTEXT,
    #     prediction_column_suffix='few_shot_cot',
    #     model_name=GEMINI_MODEL_NAME
    # )
    # 失敗した予測の再試行の例
    # retry_llm_predictions(
    #     column_to_retry='prediction_rule3_gemini_1_5_flash_zeroshot', # 再試行したいカラム名
    #     prompt_file_path=PROMPT_FILE_ZERO_SHOT_FULLTEXT,
    #     text_column='full_text'
    # )
