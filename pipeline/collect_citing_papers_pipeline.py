import pandas as pd
import os
from src.collect_citing_papers import (
    load_data_papers_for_citing_collection,
    list_citing_papers,
    download_citing_papers_xml,
    save_citing_papers_results,
    retry_failed_downloads
)
from src.config import (
    SCOPUS_API_KEY, OUTPUT_FILE_DATA_PAPERS, OUTPUT_FILE_CITING_PAPERS_WITH_PATHS,
    XML_OUTPUT_DIR
)

def run_collect_citing_papers_pipeline(
    api_key: str = SCOPUS_API_KEY,
    input_data_papers_csv: str = OUTPUT_FILE_DATA_PAPERS,
    output_citing_papers_csv: str = OUTPUT_FILE_CITING_PAPERS_WITH_PATHS,
    xml_output_dir: str = XML_OUTPUT_DIR,
    min_citations: int = 10,
    max_workers_list_citing: int = 5, # For listing citing papers (API calls)
    max_workers_download_xml: int = 10, # For downloading XMLs (parallel file ops)
    retry_failed: bool = True
):
    """
    引用論文収集パイプラインを実行します。

    Args:
        api_key (str): Scopus APIキー。
        input_data_papers_csv (str): データ論文が保存されたCSVファイルのパス。
        output_citing_papers_csv (str): 収集した引用論文の情報を保存するCSVファイルのパス。
        xml_output_dir (str): ダウンロードしたXMLファイルを保存するディレクトリのパス。
        min_citations (int): 処理対象とするデータ論文の最小被引用数。
        max_workers_list_citing (int): 引用論文リストアップ時の並列処理スレッド数。
        max_workers_download_xml (int): XMLダウンロード時の並列処理スレッド数。
        retry_failed (bool): 失敗したXMLダウンロードを再試行するかどうか。
    """
    print("--- 引用論文収集フェーズ開始 ---")

    # 1. データ論文の読み込みとフィルタリング
    df_target_data_papers = load_data_papers_for_citing_collection(
        input_csv=input_data_papers_csv,
        min_citations=min_citations
    )
    if df_target_data_papers.empty:
        print("処理対象のデータ論文がないため、引用論文収集をスキップします。")
        print("--- 引用論文収集フェーズ完了 ---")
        return

    # 2. 被引用論文のリストアップ
    # Note: list_citing_papers internally handles API calls and rate limits.
    # max_workers_list_citing is not directly used here as the original function
    # iterates sequentially per data paper, but it's good to keep as a parameter
    # for potential future parallelization of the outer loop.
    citing_papers_list = list_citing_papers(
        df_data_papers=df_target_data_papers,
        api_key=api_key
    )
    if not citing_papers_list:
        print("リストアップされた被引用論文がないため、XMLダウンロードをスキップします。")
        print("--- 引用論文収集フェーズ完了 ---")
        return

    # 3. 全文XMLの並列ダウンロード
    df_download_results = download_citing_papers_xml(
        tasks=citing_papers_list,
        api_key=api_key,
        output_dir=xml_output_dir,
        max_workers=max_workers_download_xml
    )

    # 4. 結果の保存
    save_citing_papers_results(df_download_results, output_csv=output_citing_papers_csv)

    # 5. 失敗したダウンロードの再試行
    if retry_failed:
        retry_failed_downloads(
            input_csv=output_citing_papers_csv,
            api_key=api_key,
            output_dir=xml_output_dir,
            max_workers=max_workers_download_xml // 2 # 再試行時はスレッド数を減らす
        )
    
    print("--- 引用論文収集フェーズ完了 ---")

if __name__ == "__main__":
    # このスクリプトを直接実行する場合の例
    run_collect_citing_papers_pipeline()
