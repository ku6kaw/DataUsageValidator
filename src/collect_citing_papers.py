import requests
import pandas as pd
import time
import os
from tqdm import tqdm
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
import random # For exponential backoff in retries

from src.config import (
    SCOPUS_API_KEY, SCOPUS_BASE_URL, SCOPUS_FULLTEXT_API_EID_URL, SCOPUS_FULLTEXT_API_DOI_URL,
    OUTPUT_FILE_DATA_PAPERS, OUTPUT_FILE_CITING_PAPERS_RAW, OUTPUT_FILE_CITING_PAPERS_WITH_PATHS,
    XML_OUTPUT_DIR
)

def load_data_papers_for_citing_collection(input_csv: str = OUTPUT_FILE_DATA_PAPERS, min_citations: int = 10) -> pd.DataFrame:
    """
    データ論文CSVを読み込み、被引用数でフィルタリングして、引用論文収集の対象を準備します。

    Args:
        input_csv (str): データ論文CSVファイルのパス。
        min_citations (int): 処理対象とする最小被引用数。

    Returns:
        pd.DataFrame: フィルタリングされたデータ論文のDataFrame。
    """
    try:
        df_data_papers = pd.read_csv(input_csv)
        df_target = df_data_papers.dropna(subset=['eid']).copy()
        df_target['citedby_count'] = pd.to_numeric(df_target['citedby_count'], errors='coerce').fillna(0)
        df_target = df_target[df_target['citedby_count'] >= min_citations].copy()
        df_target = df_target.sort_values(by='citedby_count', ascending=False).reset_index(drop=True)
        print(f"処理対象のデータ論文（被引用数{min_citations}以上）: {len(df_target)}件")
        return df_target
    except FileNotFoundError:
        print(f"エラー: '{input_csv}' が見つかりません。データ論文収集ノートブックを先に実行してください。")
        return pd.DataFrame()

def list_citing_papers(df_data_papers: pd.DataFrame, api_key: str = SCOPUS_API_KEY) -> list:
    """
    データ論文を引用している論文のメタデータをScopus APIからリストアップします。

    Args:
        df_data_papers (pd.DataFrame): 引用論文を検索する対象のデータ論文DataFrame。
        api_key (str): Scopus APIキー。

    Returns:
        list: 収集した被引用論文のメタデータ辞書のリスト。
    """
    tasks = []
    if df_data_papers.empty:
        print("データ論文がありません。被引用論文のリストアップをスキップします。")
        return tasks

    print("\n[ステップA] ダウンロード対象の被引用論文リストを作成しています...")
    
    for index, data_paper in tqdm(df_data_papers.iterrows(), total=len(df_data_papers), desc="データ論文をスキャン中"):
        data_paper_eid = data_paper['eid']
        data_paper_title = data_paper['title']
        
        search_params = {
            'apiKey': api_key, 'query': f"REF({data_paper_eid})",
            'cursor': '*', 'count': 25, 'view': 'STANDARD'
        }
        
        while 'cursor' in search_params and search_params['cursor']:
            try:
                search_response = requests.get(SCOPUS_BASE_URL, params=search_params)
                search_response.raise_for_status()
                search_data = search_response.json()
                
                entries = search_data.get('search-results', {}).get('entry', [])
                if not entries:
                    break
                
                for entry in entries:
                    tasks.append({
                        'citing_paper_eid': entry.get('eid'),
                        'citing_paper_doi': entry.get('prism:doi'),
                        'citing_paper_title': entry.get('dc:title'),
                        'citing_paper_year': entry.get('prism:coverDate', '')[:4],
                        'cited_data_paper_title': data_paper_title,
                    })

                next_cursor_url = next((link.get('@href') for link in search_data.get('search-results', {}).get('link', []) if link.get('@ref') == 'next'), None)
                
                if next_cursor_url:
                    parsed_url = urlparse(next_cursor_url)
                    query_params = parse_qs(parsed_url.query)
                    search_params['cursor'] = query_params.get('cursor', [None])[0]
                else:
                    break
                
                time.sleep(1)

            except requests.exceptions.RequestException as e:
                print(f"  - API検索中にエラー (EID: {data_paper_eid}): {e}")
                break
    
    print(f"[ステップA] 完了。合計 {len(tasks)} 件の被引用論文をリストアップしました。")
    return tasks

def sanitize_filename(filename: str) -> str:
    """ファイル名として使えない文字をアンダースコアに置換する"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def download_xml_by_doi(task: dict, api_key: str = SCOPUS_API_KEY, output_dir: str = XML_OUTPUT_DIR, max_retries: int = 3) -> dict:
    """
    個々の論文の全文XMLをDOIを使ってScienceDirect APIからダウンロードします。
    レートリミット(429)エラーに対しては指数バックオフでリトライします。

    Args:
        task (dict): 論文情報を含む辞書（'citing_paper_doi'が必要）。
        api_key (str): ScienceDirect APIキー。
        output_dir (str): XMLファイルを保存するディレクトリ。
        max_retries (int): 最大リトライ回数。

    Returns:
        dict: ダウンロード結果とステータスが追加された論文情報辞書。
    """
    doi = task.get('citing_paper_doi')
    if not doi or pd.isna(doi):
        task['fulltext_xml_path'] = None
        task['download_status'] = "failed (DOI is missing)"
        return task

    safe_filename = sanitize_filename(doi) + '.xml'
    xml_path = os.path.join(output_dir, safe_filename)
    
    if os.path.exists(xml_path):
        task['fulltext_xml_path'] = xml_path
        task['download_status'] = "success (cached)"
        return task

    for attempt in range(max_retries):
        try:
            url = f"{SCOPUS_FULLTEXT_API_DOI_URL}{doi}?apiKey={api_key}"
            response = requests.get(url, headers={'Accept': 'application/xml'}, timeout=60)
            
            if response.status_code == 200:
                with open(xml_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                task['fulltext_xml_path'] = xml_path
                task['download_status'] = f"success (downloaded, attempt {attempt + 1})"
                return task
            elif response.status_code == 429:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"  - レートリミット (DOI: {doi})。{wait_time:.2f}秒待機してリトライします (試行 {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                task['fulltext_xml_path'] = None
                task['download_status'] = f"failed (Status: {response.status_code})"
                return task
        except requests.exceptions.RequestException as e:
            print(f"  - リクエストエラー (DOI: {doi}, 試行 {attempt + 1}/{max_retries}): {e}")
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait_time)
    
    task['fulltext_xml_path'] = None
    task['download_status'] = f"failed (retries exhausted)"
    return task

def download_citing_papers_xml(tasks: list, api_key: str = SCOPUS_API_KEY, output_dir: str = XML_OUTPUT_DIR, max_workers: int = 10) -> pd.DataFrame:
    """
    リストアップされた被引用論文のXMLを並列でダウンロードします。

    Args:
        tasks (list): 被引用論文のメタデータ辞書のリスト。
        api_key (str): ScienceDirect APIキー。
        output_dir (str): XMLファイルを保存するディレクトリ。
        max_workers (int): 並列ダウンロードに使用するスレッド数。

    Returns:
        pd.DataFrame: ダウンロード結果が更新されたDataFrame。
    """
    if not tasks:
        print("ダウンロード対象の被引用論文がありません。")
        return pd.DataFrame()

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[ステップB] {len(tasks)}件の論文XMLのダウンロードを並列で開始します...")
    
    results_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # download_xml_by_doi関数にapi_keyとoutput_dirを部分適用
        func = lambda task: download_xml_by_doi(task, api_key=api_key, output_dir=output_dir)
        results_list = list(tqdm(executor.map(func, tasks), total=len(tasks), desc="XMLダウンロード中"))

    df_results = pd.DataFrame(results_list)
    print(f"\nダウンロード処理完了。")
    print("\n--- 処理結果サマリー ---")
    print(df_results['download_status'].value_counts())
    return df_results

def save_citing_papers_results(df: pd.DataFrame, output_csv: str = OUTPUT_FILE_CITING_PAPERS_WITH_PATHS):
    """
    被引用論文の収集結果をCSVファイルに保存します。

    Args:
        df (pd.DataFrame): 保存するDataFrame。
        output_csv (str): 保存先のCSVファイルパス。
    """
    if not df.empty:
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n処理完了。結果を '{output_csv}' に保存しました。")
        print("\n--- 保存されたデータの先頭5件 ---")
        print(df.head())
    else:
        print("保存するデータがありませんでした。")

def retry_failed_downloads(input_csv: str = OUTPUT_FILE_CITING_PAPERS_WITH_PATHS, api_key: str = SCOPUS_API_KEY, output_dir: str = XML_OUTPUT_DIR, max_workers: int = 2):
    """
    以前のダウンロードで失敗した（特に429エラー）論文のXMLダウンロードを再試行します。

    Args:
        input_csv (str): 以前のダウンロード結果が保存されたCSVファイルのパス。
        api_key (str): ScienceDirect APIキー。
        output_dir (str): XMLファイルを保存するディレクトリ。
        max_workers (int): 並列ダウンロードに使用するスレッド数（再試行時は少なめに設定推奨）。
    """
    try:
        df_results = pd.read_csv(input_csv)
        retry_targets_df = df_results[df_results['download_status'].str.contains('failed', na=False)].copy()
        
        if not retry_targets_df.empty:
            print(f"\n失敗した {len(retry_targets_df)} 件のダウンロードを再試行します。")
            tasks_to_retry = retry_targets_df.to_dict('records')

            df_retry_results = download_citing_papers_xml(tasks_to_retry, api_key=api_key, output_dir=output_dir, max_workers=max_workers)
            
            # 元のDataFrameを更新
            df_results.set_index('citing_paper_doi', inplace=True)
            df_retry_results.set_index('citing_paper_doi', inplace=True)
            df_results.update(df_retry_results)
            df_results.reset_index(inplace=True)
            
            save_citing_papers_results(df_results, output_csv=input_csv)
            print("\n--- 最新の処理結果サマリー ---")
            print(df_results['download_status'].value_counts())
        else:
            print("再試行する失敗したタスクはありませんでした。")
            
    except FileNotFoundError:
        print(f"エラー: '{input_csv}' が見つかりません。")

def main_collect_citing_papers():
    """
    引用論文収集のメイン処理を実行します。
    """
    # 1. データ論文の読み込みとフィルタリング
    df_target_data_papers = load_data_papers_for_citing_collection()

    # 2. 被引用論文のリストアップ
    citing_papers_list = list_citing_papers(df_target_data_papers)

    # 3. 全文XMLの並列ダウンロード
    df_final_results = download_citing_papers_xml(citing_papers_list)

    # 4. 結果の保存
    save_citing_papers_results(df_final_results)

if __name__ == "__main__":
    main_collect_citing_papers()
