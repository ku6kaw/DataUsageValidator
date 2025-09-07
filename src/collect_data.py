import pandas as pd
import os
from src.scopus_api import get_total_data_papers_count, collect_data_papers, save_data_papers_to_csv
from src.config import SCOPUS_API_KEY, SCOPUS_QUERY_DATA_PAPERS, OUTPUT_FILE_DATA_PAPERS, OUTPUT_DIR_PROCESSED

def main():
    """
    データ論文収集のメイン処理を実行します。
    """
    # 1. 総件数の取得
    total_results = get_total_data_papers_count(api_key=SCOPUS_API_KEY, query=SCOPUS_QUERY_DATA_PAPERS)

    if total_results > 0:
        # 2. データ論文の収集
        df_data_papers = collect_data_papers(api_key=SCOPUS_API_KEY, query=SCOPUS_QUERY_DATA_PAPERS, total_results=total_results)

        # 3. データの保存
        if not df_data_papers.empty:
            save_data_papers_to_csv(df_data_papers, output_file=OUTPUT_FILE_DATA_PAPERS, output_dir=OUTPUT_DIR_PROCESSED)
        else:
            print("収集できたデータがありませんでした。")
    else:
        print("収集対象のデータ論文がありませんでした。")

if __name__ == "__main__":
    main()
