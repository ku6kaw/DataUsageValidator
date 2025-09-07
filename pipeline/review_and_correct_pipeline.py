import pandas as pd
import os
from src.review_and_correction import (
    load_and_merge_review_data,
    identify_disagreements,
    generate_review_prompts,
    apply_corrections_to_ground_truth
)
from src.config import (
    OUTPUT_FILE_ANNOTATION_TARGET_LIST,
    OUTPUT_FILE_PREDICTION_LLM,
    OUTPUT_FILE_SAMPLES_WITH_TEXT
)

def run_review_and_correction_pipeline(
    ground_truth_csv: str = OUTPUT_FILE_ANNOTATION_TARGET_LIST,
    llm_predictions_csv: str = OUTPUT_FILE_PREDICTION_LLM,
    samples_with_text_csv: str = OUTPUT_FILE_SAMPLES_WITH_TEXT,
    best_model_column: str = 'prediction_rule3_gemini-2_5-flash', # 評価で最も良かったモデルのカラムを指定
    corrections: dict = None, # 手動修正を適用する場合の辞書 {doi: new_label}
    generate_prompts: bool = True # 食い違いがあった場合にレビュープロンプトを生成するかどうか
):
    """
    レビューと修正パイプラインを実行します。
    人間の判断とLLMの判断の食い違いを特定し、レビュープロンプトを生成、
    必要に応じて正解データに手動修正を適用します。

    Args:
        ground_truth_csv (str): 正解データCSVファイルのパス。
        llm_predictions_csv (str): LLM予測結果CSVファイルのパス。
        samples_with_text_csv (str): テキスト抽出済みデータCSVファイルのパス。
        best_model_column (str): 評価対象のLLM予測結果カラム名。
        corrections (dict): DOIをキー、新しいラベル（0または1）を値とする辞書。
                            この辞書がNoneでない場合、修正が適用されます。
        generate_prompts (bool): 食い違いがあった場合にレビュープロンプトを生成するかどうか。
    """
    print("--- レビューと修正フェーズ開始 ---")

    # 1. レビューデータの読み込みと結合
    df_review = load_and_merge_review_data(
        ground_truth_csv=ground_truth_csv,
        llm_predictions_csv=llm_predictions_csv,
        samples_with_text_csv=samples_with_text_csv,
        best_model_column=best_model_column
    )
    if df_review.empty:
        print("レビュー対象のデータがないため、レビューと修正をスキップします。")
        print("--- レビューと修正フェーズ完了 ---")
        return

    # 2. 人間の判断とLLMの判断の食い違いを特定
    disagreements = identify_disagreements(df_review, best_model_column)

    # 3. レビュープロンプトの生成
    if generate_prompts:
        generate_review_prompts(disagreements, best_model_column) # best_model_columnを渡す

    # 4. 手動修正の適用
    if corrections:
        apply_corrections_to_ground_truth(corrections, ground_truth_csv)
    
    print("--- レビューと修正フェーズ完了 ---")

if __name__ == "__main__":
    # このスクリプトを直接実行する場合の例
    # 食い違いの特定とプロンプト生成
    run_review_and_correction_pipeline(
        best_model_column='prediction_rule3_gemini-2_5-flash',
        generate_prompts=True,
        corrections=None
    )

    # 手動修正を適用する例 (必要に応じてコメントを外して実行)
    # corrections_to_apply = {
    #     "10.1016/j.jprot.2022.104578": 1, # このDOIのラベルを1に修正
    #     "10.1016/j.ccell.2021.04.015": 0  # このDOIのラベルを0に修正
    # }
    # run_review_and_correction_pipeline(
    #     best_model_column='prediction_rule3_gemini-2_5-flash',
    #     generate_prompts=False, # 修正適用時はプロンプト生成をスキップしても良い
    #     corrections=corrections_to_apply
    # )
