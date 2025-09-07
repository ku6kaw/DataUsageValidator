import pandas as pd
import numpy as np
import os
from src.config import OUTPUT_FILE_DATA_PAPERS, OUTPUT_DIR_PROCESSED

def load_and_preprocess_data_papers(data_file: str = OUTPUT_FILE_DATA_PAPERS) -> pd.DataFrame:
    """
    データ論文のCSVファイルを読み込み、前処理（citedby_countの数値変換）を行います。

    Args:
        data_file (str): データ論文CSVファイルのパス。

    Returns:
        pd.DataFrame: 前処理されたデータ論文のDataFrame。ファイルが見つからない場合は空のDataFrameを返します。
    """
    try:
        df = pd.read_csv(data_file)
        print(f"'{data_file}' を正常に読み込みました。")
        print(f"データ論文の総数: {len(df)}件")

        # citedby_count列を数値型に変換（エラーはNaNにし、その後0で埋める）
        df['citedby_count'] = pd.to_numeric(df['citedby_count'], errors='coerce').fillna(0)
        return df
    except FileNotFoundError:
        print(f"エラー: '{data_file}' が見つかりません。")
        print("先にデータ収集ノートブックを実行してください。")
        return pd.DataFrame()

def filter_by_citation_count(df: pd.DataFrame, min_citations: int = 2) -> pd.DataFrame:
    """
    被引用数が指定された閾値以上のデータ論文に絞り込みます。

    Args:
        df (pd.DataFrame): 元のデータ論文のDataFrame。
        min_citations (int): 最小被引用数。

    Returns:
        pd.DataFrame: フィルタリングされたデータ論文のDataFrame。
    """
    if df.empty:
        return pd.DataFrame()
    
    df_filtered = df[df['citedby_count'] >= min_citations].copy()
    print(f"被引用数が{min_citations}以上のデータ論文の数: {len(df_filtered)}件")
    return df_filtered

def analyze_citation_statistics(df: pd.DataFrame, title: str = "被引用数の基本統計量"):
    """
    データ論文の被引用数に関する基本統計量を表示します。

    Args:
        df (pd.DataFrame): 分析対象のデータ論文DataFrame。
        title (str): 統計量表示のタイトル。
    """
    if not df.empty:
        print(f"\n--- {title} ---")
        print(df['citedby_count'].describe(percentiles=[.25, .5, .75, .9, .95, .99]))
    else:
        print(f"\n表示対象のデータがありません。({title})")

def categorize_citations(df: pd.DataFrame) -> pd.DataFrame:
    """
    被引用数に基づいてデータ論文をカテゴリに分類します。

    Args:
        df (pd.DataFrame): 分析対象のデータ論文DataFrame。

    Returns:
        pd.DataFrame: 'citation_category'列が追加されたDataFrame。
    """
    if df.empty:
        return pd.DataFrame()

    bins = [
        1,           # 2から始まるための下限
        10,          # 75パーセンタイル(13)より少し下のキリの良い数字
        50,          # 95パーセンタイル(46)より少し上のキリの良い数字
        150,         # 99パーセンタイル(125)より少し上のキリの良い数字
        float('inf') # 151以上はすべて
    ]

    labels = [
        '2-10 (Low)',
        '11-50 (Medium)',
        '51-150 (High)',
        '151+ (Top Tier)'
    ]

    df['citation_category'] = pd.cut(df['citedby_count'], bins=bins, labels=labels, right=True)

    print("--- カテゴリごとの論文数 ---")
    print(df['citation_category'].value_counts().sort_index())
    return df
