import pandas as pd
import os
from src.config import OUTPUT_FILE_CITING_PAPERS_WITH_PATHS

def load_citing_papers_results(results_csv_path: str = OUTPUT_FILE_CITING_PAPERS_WITH_PATHS) -> pd.DataFrame:
    """
    引用論文のダウンロード結果CSVファイルを読み込みます。

    Args:
        results_csv_path (str): 結果CSVファイルのパス。

    Returns:
        pd.DataFrame: 読み込まれたDataFrame。ファイルが見つからない場合は空のDataFrameを返します。
    """
    try:
        df_results = pd.read_csv(results_csv_path)
        print(f"'{results_csv_path}' を正常に読み込みました。")
        print(f"記録されている論文総数: {len(df_results)}件")
        return df_results
    except FileNotFoundError:
        print(f"エラー: '{results_csv_path}' が見つかりません。")
        print("先に引用論文収集ノートブックを実行してください。")
        return pd.DataFrame()

def summarize_download_status(df_results: pd.DataFrame):
    """
    ダウンロード結果のステータスをサマリー表示します。

    Args:
        df_results (pd.DataFrame): ダウンロード結果のDataFrame。
    """
    if not df_results.empty:
        print("\n--- ダウンロード結果のサマリー ---")
        status_counts = df_results['download_status'].value_counts()
        print(status_counts)
    else:
        print("ダウンロード結果データがありません。")

def verify_xml_file_existence(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    ダウンロード成功と記録されたXMLファイルが実際に存在するかを検証します。

    Args:
        df_results (pd.DataFrame): ダウンロード結果のDataFrame。

    Returns:
        pd.DataFrame: ファイル存在チェックの結果が追加されたDataFrame。
    """
    if df_results.empty:
        print("ダウンロード結果データがありません。ファイル整合性チェックをスキップします。")
        return pd.DataFrame()

    print("\n--- 保存されたXMLファイルの整合性チェック ---")
    
    df_success = df_results[
        df_results['download_status'].str.startswith('success', na=False)
    ].copy()

    if not df_success.empty:
        print(f"ダウンロード成功と記録されている {len(df_success)}件について、ファイルが存在するか確認します...")
        
        df_success['file_exists'] = df_success['fulltext_xml_path'].apply(
            lambda path: os.path.exists(path) if isinstance(path, str) else False
        )
        
        existence_counts = df_success['file_exists'].value_counts()
        print("\n[チェック結果]")
        print(existence_counts)
        
        missing_files = df_success[df_success['file_exists'] == False]
        
        if not missing_files.empty:
            print("\n【警告】以下の論文は成功と記録されていますが、XMLファイルが見つかりませんでした：")
            print(missing_files[['citing_paper_doi', 'fulltext_xml_path', 'download_status']])
        else:
            print("\n[OK] 全ての成功レコードについて、ファイルが正しく存在することを確認しました。")
    else:
        print("ダウンロードに成功した論文はまだありません。")
    
    return df_success

def main_verify_collection():
    """
    引用論文収集の検証のメイン処理を実行します。
    """
    df_results = load_citing_papers_results()
    if not df_results.empty:
        summarize_download_status(df_results)
        verify_xml_file_existence(df_results)

if __name__ == "__main__":
    main_verify_collection()
