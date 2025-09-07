import pandas as pd
import os
from src.config import OUTPUT_FILE_ANNOTATION_TARGET_LIST, OUTPUT_FILE_SAMPLES_WITH_TEXT

def audit_dataframe(df: pd.DataFrame, file_name: str):
    """
    データフレームの基本情報と欠損値を監査レポートとして表示します。

    Args:
        df (pd.DataFrame): 監査対象のDataFrame。
        file_name (str): 監査対象のファイル名（表示用）。
    """
    if df.empty:
        print(f"監査対象のデータフレームが空です。ファイル: '{file_name}'")
        return

    print(f"--- 監査レポート: '{file_name}' ---")

    print("\n" + "="*50)
    print("1. データフレームの基本情報 (info)")
    print("="*50)
    df.info()

    print("\n" + "="*50)
    print("2. 各列の欠損値（null）の数")
    print("="*50)
    print(df.isnull().sum())

def compare_sample_lists(
    df_targets: pd.DataFrame, 
    df_samples: pd.DataFrame,
    target_list_name: str = "サンプリングリスト",
    samples_list_name: str = "抽出済みファイル"
):
    """
    2つのデータフレーム（サンプリングリストと抽出済みファイル）を比較し、不一致を報告します。

    Args:
        df_targets (pd.DataFrame): 比較元となるサンプリングリストのDataFrame。
        df_samples (pd.DataFrame): 比較対象となる抽出済みファイルのDataFrame。
        target_list_name (str): サンプリングリストの表示名。
        samples_list_name (str): 抽出済みファイルの表示名。
    """
    if df_targets.empty or df_samples.empty:
        print("比較対象のデータフレームが空のため、照合をスキップします。")
        return

    print("\n" + "="*50)
    print("【2ファイルの照合レポート】")
    print("="*50)

    target_dois = set(df_targets['citing_paper_doi'].unique())
    sample_dois = set(df_samples['citing_paper_doi'].unique())

    print(f"1. 件数の確認")
    print(f"   - {target_list_name}のユニークDOI数: {len(target_dois)} 件")
    print(f"   - {samples_list_name}のユニークDOI数:   {len(sample_dois)} 件")

    missing_dois = target_dois - sample_dois

    print("\n2. 不一致の特定")
    if missing_dois:
        print(f"   - ⚠️ {len(missing_dois)} 件の論文が、{samples_list_name}から欠落していることが判明しました。")
        
        df_missing = df_targets[df_targets['citing_paper_doi'].isin(missing_dois)]
        
        print("\n--- 欠落していた論文の詳細 ---")
        print(df_missing.to_string())
    else:
        print(f"   - ✅ ファイル間のDOIに不一致はありませんでした。")
        if len(target_dois) != len(sample_dois):
            print("   - ⚠️ ただし、件数が異なります。重複行などが原因の可能性があります。")

def main_check_samples():
    """
    サンプルチェックのメイン処理を実行します。
    """
    # 1. テキスト抽出済みファイルの監査
    try:
        df_samples_with_text = pd.read_csv(OUTPUT_FILE_SAMPLES_WITH_TEXT)
        audit_dataframe(df_samples_with_text, os.path.basename(OUTPUT_FILE_SAMPLES_WITH_TEXT))
    except FileNotFoundError:
        print(f"エラー: '{OUTPUT_FILE_SAMPLES_WITH_TEXT}' が見つかりません。")
        df_samples_with_text = pd.DataFrame()
    except Exception as e:
        print(f"テキスト抽出済みファイルの読み込み中にエラーが発生しました: {e}")
        df_samples_with_text = pd.DataFrame()

    # 2. 2ファイルの照合
    try:
        df_annotation_targets = pd.read_csv(OUTPUT_FILE_ANNOTATION_TARGET_LIST)
    except FileNotFoundError:
        print(f"エラー: '{OUTPUT_FILE_ANNOTATION_TARGET_LIST}' が見つかりません。")
        df_annotation_targets = pd.DataFrame()
    except Exception as e:
        print(f"アノテーションリストの読み込み中にエラーが発生しました: {e}")
        df_annotation_targets = pd.DataFrame()

    compare_sample_lists(df_annotation_targets, df_samples_with_text)

if __name__ == "__main__":
    main_check_samples()
