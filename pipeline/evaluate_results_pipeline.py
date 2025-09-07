import pandas as pd
import os
from src.evaluation import (
    load_and_merge_evaluation_data,
    generate_hybrid_predictions,
    calculate_metrics,
    save_evaluation_results
)
from src.config import (
    OUTPUT_FILE_ANNOTATION_TARGET_LIST,
    OUTPUT_FILE_FEATURES_FOR_EVALUATION,
    OUTPUT_FILE_PREDICTION_LLM,
    TABLES_DIR
)

def run_evaluate_results_pipeline(
    ground_truth_csv: str = OUTPUT_FILE_ANNOTATION_TARGET_LIST,
    features_csv: str = OUTPUT_FILE_FEATURES_FOR_EVALUATION,
    llm_predictions_csv: str = OUTPUT_FILE_PREDICTION_LLM,
    output_metrics_file_name: str = 'evaluation_metrics_summary.csv'
):
    """
    評価と分析パイプラインを実行します。
    正解データ、特徴量、LLM予測結果を結合し、各種評価指標を計算・保存します。

    Args:
        ground_truth_csv (str): 正解データCSVファイルのパス。
        features_csv (str): ルールベースの特徴量CSVファイルのパス。
        llm_predictions_csv (str): LLM予測結果CSVファイルのパス。
        output_metrics_file_name (str): 評価指標を保存するCSVファイル名。
    """
    print("--- 評価と分析フェーズ開始 ---")

    # 1. 評価データの読み込みと結合
    df_eval_base = load_and_merge_evaluation_data(
        ground_truth_csv=ground_truth_csv,
        features_csv=features_csv,
        llm_predictions_csv=llm_predictions_csv
    )
    if df_eval_base.empty:
        print("評価対象のデータがないため、評価と分析をスキップします。")
        print("--- 評価と分析フェーズ完了 ---")
        return

    # 2. ハイブリッド予測の生成
    df_eval_with_hybrid = generate_hybrid_predictions(df_eval_base)

    # 3. 評価指標の計算
    rule_columns = {
        "Rule 1 (Mention Count)": "prediction_rule1",
        "Rule 2 (Section Keyword)": "prediction_rule2",
        "Rule 3 (LLM 1.5, Zero-shot)": "prediction_rule3_fulltext",
        "Rule 3 (LLM 1.5, Few-shot)": "prediction_rule3_fulltext_few_shot",
        "Rule 3 (LLM 2.5, Few-shot)": "prediction_rule3_gemini-2_5-flash",
        "Rule 3 (LLM 2.5, Zero-shot)": "prediction_rule3_gemini-2_5-flash_zeroshot",
        "Hybrid (Rule 2 AND 2.5 Zero-shot)": "prediction_hybrid_AND_gemini2_5_zeroshot",
        "Hierarchical Hybrid Model": "prediction_hierarchical_hybrid"
    }
    
    df_metrics = calculate_metrics(df_eval_with_hybrid, rule_columns)

    # 4. 評価結果の保存と表示
    save_evaluation_results(df_metrics, output_file_name)
    
    print("--- 評価と分析フェーズ完了 ---")

if __name__ == "__main__":
    # このスクリプトを直接実行する場合の例
    run_evaluate_results_pipeline()
