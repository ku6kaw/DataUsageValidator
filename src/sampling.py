import pandas as pd
import os
from src.config import OUTPUT_FILE_CITING_PAPERS_WITH_PATHS, OUTPUT_DIR_PROCESSED

def create_annotation_sampling_list(
    results_csv_path: str = OUTPUT_FILE_CITING_PAPERS_WITH_PATHS,
    output_dir: str = 'data/ground_truth',
    output_file_name: str = 'annotation_target_list.csv',
    sample_size: int = 200,
    random_state: int = 42
) -> pd.DataFrame:
    """
    ダウンロードに成功した引用論文から、アノテーション用のサンプルリストを作成します。

    Args:
        results_csv_path (str): 論文リストが記録されたCSVファイルのパス。
        output_dir (str): 出力先のディレクトリ。
        output_file_name (str): 出力ファイル名。
        sample_size (int): 抽出したいサンプル数。
        random_state (int): ランダムサンプリングの再現性を保証するためのシード。

    Returns:
        pd.DataFrame: サンプリングされた論文のDataFrame。
    """
    output_file_path = os.path.join(output_dir, output_file_name)

    try:
        df_results = pd.read_csv(results_csv_path)
        
        # 本文XMLのダウンロードに成功した（cached含む）論文のみをサンプリングの母集団とする
        df_success = df_results[
            df_results['download_status'].str.startswith('success', na=False)
        ].copy()
        
        print(f"XML取得に成功した {len(df_success)} 件の論文からサンプリングを行います。")

    except FileNotFoundError:
        print(f"エラー: '{results_csv_path}' が見つかりません。")
        print("先に引用論文収集ノートブックを実行してください。")
        return pd.DataFrame()

    if not df_success.empty and len(df_success) >= sample_size:
        sample_df = df_success.sample(n=sample_size, random_state=random_state)
        
        columns_to_save = [
            'citing_paper_eid', 
            'citing_paper_doi', 
            'citing_paper_title',
            'cited_data_paper_title'
        ]
        
        os.makedirs(output_dir, exist_ok=True)
        sample_df[columns_to_save].to_csv(output_file_path, index=False, encoding='utf-8-sig')
        
        print(f"\nサンプリング完了。")
        print(f"アノテーション対象 {sample_size} 件のリストを '{output_file_path}' に保存しました。")
        
        print("\n--- サンプルリストの先頭5件 ---")
        print(sample_df[columns_to_save].head())
        return sample_df[columns_to_save]
    else:
        print(f"サンプリング対象となる、ダウンロードに成功した論文が{sample_size}件未満です。")
        return pd.DataFrame()

if __name__ == "__main__":
    # このスクリプトを直接実行した場合の例
    create_annotation_sampling_list()
