import pandas as pd
import os
from src.sampling import create_annotation_sampling_list
from src.data_processor import extract_text_from_xml_files
from src.config import (
    OUTPUT_FILE_CITING_PAPERS_WITH_PATHS,
    OUTPUT_FILE_ANNOTATION_TARGET_LIST,
    OUTPUT_FILE_SAMPLES_WITH_TEXT,
    OUTPUT_DIR_PROCESSED
)

def run_prepare_data_pipeline(
    citing_papers_master_csv: str = OUTPUT_FILE_CITING_PAPERS_WITH_PATHS,
    annotation_target_list_csv: str = OUTPUT_FILE_ANNOTATION_TARGET_LIST,
    samples_with_text_csv: str = OUTPUT_FILE_SAMPLES_WITH_TEXT,
    processed_output_dir: str = OUTPUT_DIR_PROCESSED,
    sample_size: int = 200,
    random_state: int = 42
):
    """
    データ準備パイプラインを実行します。
    サンプリング、XMLからのテキスト抽出が含まれます。

    Args:
        citing_papers_master_csv (str): 全ての引用論文のマスターリストCSVパス（XMLパス情報を含む）。
        annotation_target_list_csv (str): アノテーション対象の論文リストCSVパス。
        samples_with_text_csv (str): 抽出されたテキストが追加されたサンプルを保存するCSVパス。
        processed_output_dir (str): 処理済みデータを保存するディレクトリのパス。
        sample_size (int): 抽出したいサンプル数。
        random_state (int): ランダムサンプリングの再現性を保証するためのシード。
    """
    print("--- データ準備フェーズ開始 ---")

    # 1. アノテーション用サンプリングリストの作成
    df_annotation_targets = create_annotation_sampling_list(
        results_csv_path=citing_papers_master_csv,
        output_dir=os.path.dirname(annotation_target_list_csv), # configからディレクトリを抽出
        output_file_name=os.path.basename(annotation_target_list_csv), # configからファイル名を抽出
        sample_size=sample_size,
        random_state=random_state
    )
    if df_annotation_targets.empty:
        print("アノテーション対象の論文がないため、テキスト抽出をスキップします。")
        print("--- データ準備フェーズ完了 ---")
        return

    # 2. XMLファイルからのテキスト抽出
    df_samples_with_text = extract_text_from_xml_files(
        annotation_list_csv=annotation_target_list_csv,
        master_list_csv=citing_papers_master_csv,
        output_dir=processed_output_dir,
        output_file_name=os.path.basename(samples_with_text_csv)
    )
    if df_samples_with_text.empty:
        print("テキストが抽出されたサンプルがないため、データ準備を完了します。")
    
    print("--- データ準備フェーズ完了 ---")

if __name__ == "__main__":
    # このスクリプトを直接実行する場合の例
    run_prepare_data_pipeline()
