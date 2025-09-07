import requests
import pandas as pd
import time
import os
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm
from src.config import SCOPUS_API_KEY, SCOPUS_BASE_URL, SCOPUS_QUERY_DATA_PAPERS, OUTPUT_DIR_PROCESSED, OUTPUT_FILE_DATA_PAPERS

def get_total_data_papers_count(api_key: str = SCOPUS_API_KEY, query: str = SCOPUS_QUERY_DATA_PAPERS) -> int:
    """
    Scopus APIからデータ論文の総件数を取得します。

    Args:
        api_key (str): Scopus APIキー。
        query (str): 検索クエリ。

    Returns:
        int: データ論文の総件数。エラーが発生した場合は0を返します。
    """
    print("データ論文の総件数を取得しています...")
    try:
        params = {'apiKey': api_key, 'query': query, 'count': 1}
        response = requests.get(SCOPUS_BASE_URL, params=params)
        response.raise_for_status()
        total_results = int(response.json().get('search-results', {}).get('opensearch:totalResults', 0))
        print(f"収集対象のデータ論文は全 {total_results} 件です。")
        return total_results
    except requests.exceptions.RequestException as e:
        print(f"総件数の取得中にエラーが発生しました: {e}")
        return 0

def collect_data_papers(api_key: str = SCOPUS_API_KEY, query: str = SCOPUS_QUERY_DATA_PAPERS, total_results: int = 0) -> pd.DataFrame:
    """
    Scopus APIからデータ論文の情報を収集します。

    Args:
        api_key (str): Scopus APIキー。
        query (str): 検索クエリ。
        total_results (int): 収集するデータ論文の総件数（プログレスバー用）。

    Returns:
        pd.DataFrame: 収集したデータ論文の情報を格納したDataFrame。
    """
    all_papers_data = []
    pbar = tqdm(total=total_results, desc="データ論文収集中")

    if total_results == 0:
        pbar.close()
        return pd.DataFrame()

    params = {
        'apiKey': api_key,
        'query': query,
        'cursor': '*',
        'count': 25,
        'view': 'STANDARD'
    }

    try:
        while 'cursor' in params and params['cursor']:
            response = requests.get(SCOPUS_BASE_URL, params=params)
            response.raise_for_status()

            data = response.json()
            results = data.get('search-results', {})
            entries = results.get('entry', [])

            if not entries:
                break
            
            for entry in entries:
                all_papers_data.append({
                    'eid': entry.get('eid', ''),
                    'doi': entry.get('prism:doi'),
                    'title': entry.get('dc:title'),
                    'publication_year': entry.get('prism:coverDate', '')[:4],
                    'citedby_count': entry.get('citedby-count')
                })
            
            pbar.update(len(entries))

            next_cursor_url = None
            for link in results.get('link', []):
                if link.get('@ref') == 'next':
                    next_cursor_url = link.get('@href')
                    break
            
            if next_cursor_url:
                parsed_url = urlparse(next_cursor_url)
                query_params = parse_qs(parsed_url.query)
                params['cursor'] = query_params.get('cursor', [None])[0]
            else:
                break

            time.sleep(1) # サーバーへの配慮

    except requests.exceptions.RequestException as e:
        print(f"\nAPIリクエスト中にエラーが発生しました: {e}")
    except Exception as e:
        print(f"\n予期せぬエラーが発生しました: {e}")
    finally:
        pbar.close()

    if len(all_papers_data) < total_results:
        print(f"\n注意: 処理が途中で終了した可能性があります。{total_results}件中 {len(all_papers_data)}件を収集しました。")

    return pd.DataFrame(all_papers_data)

def save_data_papers_to_csv(df: pd.DataFrame, output_file: str = OUTPUT_FILE_DATA_PAPERS, output_dir: str = OUTPUT_DIR_PROCESSED):
    """
    収集したデータ論文のDataFrameをCSVファイルに保存します。

    Args:
        df (pd.DataFrame): 保存するデータ論文のDataFrame。
        output_file (str): 保存先のファイルパス。
        output_dir (str): 保存先のディレクトリパス。
    """
    if not df.empty:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n処理完了。合計 {len(df)} 件のデータ論文を '{output_file}' に保存しました。")
        print("\n--- 保存されたデータの先頭5件 ---")
        print(df.head())
    else:
        print("収集できたデータがありませんでした。")
