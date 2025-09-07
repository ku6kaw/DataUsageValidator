import pandas as pd
import os
from src.scopus_api import get_total_data_papers_count, collect_data_papers, save_data_papers_to_csv
from src.config import SCOPUS_API_KEY, SCOPUS_QUERY_DATA_PAPERS, OUTPUT_FILE_DATA_PAPERS, OUTPUT_DIR_PROCESSED

def run_collect_data_pipeline(api_key: str = SCOPUS_API_KEY, query: str = SCOPUS_QUERY_DATA_PAPERS, output_file: str = OUTPUT_FILE_DATA_PAPERS, output_dir: str = OUTPUT_DIR_PROCESSED):
    """
    データ論文収集パイプラインを実行します。

    Args:
        api_key (str): Scopus APIキー。
        query (str): 検索クエリ。
        output_file (str): 収集したデータ論文を保存するCSVファイルのパス。
        output_dir (str): 処理済みデータを保存するディレクトリのパス。
    """
    print("--- データ論文収集フェーズ開始 ---")

    # 1. 総件数の取得
    total_results = get_total_data_papers_count(api_key=api_key, query=query)

    if total_results > 0:
        # 2. データ論文の収集
        df_data_papers = collect_data_papers(api_key=api_key, query=query, total_results=total_results)

        # 3. データの保存
        if not df_data_papers.empty:
            save_data_papers_to_csv(df_data_papers, output_file=output_file, output_dir=output_dir)
        else:
            print("収集できたデータがありませんでした。")
    else:
        print("収集対象のデータ論文がありませんでした。")
    
    print("--- データ論文収集フェーズ完了 ---")

if __name__ == "__main__":
    # このスクリプトを直接実行する場合の例
    run_collect_data_pipeline()
