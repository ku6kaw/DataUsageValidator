import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import numpy as np

from src.config import (
    OUTPUT_FILE_ANNOTATION_TARGET_LIST,
    OUTPUT_FILE_FEATURES_FOR_EVALUATION,
    OUTPUT_FILE_PREDICTION_LLM,
    RESULTS_DIR,
    TABLES_DIR
)

def load_and_merge_evaluation_data(
    ground_truth_csv: str = OUTPUT_FILE_ANNOTATION_TARGET_LIST,
    features_csv: str = OUTPUT_FILE_FEATURES_FOR_EVALUATION,
    llm_predictions_csv: str = OUTPUT_FILE_PREDICTION_LLM
) -> pd.DataFrame:
    """
    正解データ、特徴量、LLM予測結果を読み込み、結合します。

    Args:
        ground_truth_csv (str): 正解データCSVファイルのパス。
        features_csv (str): ルールベースの特徴量CSVファイルのパス。
        llm_predictions_csv (str): LLM予測結果CSVファイルのパス。

    Returns:
        pd.DataFrame: 結合された評価用DataFrame。ファイルが見つからない場合は空のDataFrameを返します。
    """
    try:
        df_gt = pd.read_csv(ground_truth_csv)
        df_features = pd.read_csv(features_csv)
        df_llm = pd.read_csv(llm_predictions_csv)

        df_eval_base = pd.merge(
            df_gt[['citing_paper_doi', 'is_data_used_gt']],
            df_features[['citing_paper_doi', 'prediction_rule1', 'prediction_rule2']],
            on='citing_paper_doi', how='inner'
        )
        
        # 評価に使う可能性のある全てのLLM結果カラムを定義
        llm_columns_to_merge = [
            'citing_paper_doi', 
            'prediction_rule3_abstract', 
            'prediction_rule3_fulltext', 
            'prediction_rule3_fulltext_few_shot',
            'prediction_rule3_gemini-2_5-flash',
            'prediction_rule3_gemini-2_5-flash_zeroshot'
        ]
        df_eval_base = pd.merge(
            df_eval_base,
            df_llm[[col for col in llm_columns_to_merge if col in df_llm.columns]],
            on='citing_paper_doi', how='left'
        )
        
        df_eval_base = df_eval_base[df_eval_base['is_data_used_gt'].isin([0, 1])].copy()
        df_eval_base['is_data_used_gt'] = df_eval_base['is_data_used_gt'].astype(int)
        
        print(f"全データを正常に読み込み・結合しました。評価対象論文数: {len(df_eval_base)}件")
        return df_eval_base

    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません。パスを確認してください。 {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"データの読み込みまたは結合中にエラーが発生しました: {e}")
        return pd.DataFrame()

def generate_hybrid_predictions(df_eval_base: pd.DataFrame) -> pd.DataFrame:
    """
    結合されたデータフレームからハイブリッド手法の予測を生成します。

    Args:
        df_eval_base (pd.DataFrame): 結合された評価用DataFrame。

    Returns:
        pd.DataFrame: ハイブリッド予測が追加されたDataFrame。
    """
    if df_eval_base.empty:
        return pd.DataFrame()

    df = df_eval_base.copy()

    # ANDゲートハイブリッド
    if 'prediction_rule3_fulltext' in df.columns:
        df['prediction_hybrid_AND_zeroshot'] = ((df['prediction_rule2'] == 1) & (df['prediction_rule3_fulltext'] == 1)).astype(int)
    if 'prediction_rule3_fulltext_few_shot' in df.columns:
        df['prediction_hybrid_AND_fewshot'] = ((df['prediction_rule2'] == 1) & (df['prediction_rule3_fulltext_few_shot'] == 1)).astype(int)
    if 'prediction_rule3_gemini-2_5-flash' in df.columns:
        df['prediction_hybrid_AND_gemini2_5'] = ((df['prediction_rule2'] == 1) & (df['prediction_rule3_gemini-2_5-flash'] == 1)).astype(int)
    if 'prediction_rule3_gemini-2_5-flash_zeroshot' in df.columns:
        df['prediction_hybrid_AND_gemini2_5_zeroshot'] = ((df['prediction_rule2'] == 1) & (df['prediction_rule3_gemini-2_5-flash_zeroshot'] == 1)).astype(int)

    # 階層的ハイブリッドモデルのロジック
    if 'prediction_rule3_gemini-2_5-flash' in df.columns and 'prediction_rule3_fulltext' in df.columns:
        default_prediction = df['prediction_rule3_gemini-2_5-flash']
        condition1 = (df['prediction_rule2'] == 1) & (df['prediction_rule3_fulltext'] == 1)
        condition2 = (df['prediction_rule3_fulltext'] == 0)
        df['prediction_hierarchical_hybrid'] = np.select(
            [condition1, condition2],
            [1, 0],
            default=default_prediction
        )
    return df

def calculate_metrics(df_eval: pd.DataFrame, rule_columns: dict) -> pd.DataFrame:
    """
    指定されたルールに基づいて評価指標を計算します。

    Args:
        df_eval (pd.DataFrame): 評価対象のDataFrame。
        rule_columns (dict): 評価するルール名と対応する予測カラム名の辞書。

    Returns:
        pd.DataFrame: 各ルールの評価指標を含むDataFrame。
    """
    if df_eval.empty:
        return pd.DataFrame()

    results = []
    
    for name, col in rule_columns.items():
        if col not in df_eval.columns:
            continue

        eval_subset = df_eval[['citing_paper_doi', 'is_data_used_gt', col]].copy()
        eval_subset.dropna(subset=[col], inplace=True)
        eval_subset = eval_subset[eval_subset[col] != -1]
        eval_subset[col] = eval_subset[col].astype(int)
        
        y_true = eval_subset['is_data_used_gt']
        y_pred = eval_subset[col]
        
        if len(y_true) == 0: continue

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        results.append({
            'Rule': name,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, zero_division=0),
            'Eval_Count': len(y_true)
        })
        
    return pd.DataFrame(results)

def save_evaluation_results(df_metrics: pd.DataFrame, output_file_name: str = 'evaluation_metrics_summary.csv'):
    """
    評価結果をCSVファイルに保存し、表示します。

    Args:
        df_metrics (pd.DataFrame): 評価指標を含むDataFrame。
        output_file_name (str): 保存するCSVファイル名。
    """
    if df_metrics.empty:
        print("評価結果データがありません。保存をスキップします。")
        return

    os.makedirs(TABLES_DIR, exist_ok=True)
    output_path = os.path.join(TABLES_DIR, output_file_name)
    df_metrics.to_csv(output_path, index=False)
    
    print(f"\n処理完了。評価結果を '{output_path}' に保存しました。")
    print("\n" + "="*80)
    print("【最終評価結果レポート】")
    print("="*80)
    print(df_metrics.round(3))

def main_analyze_results(
    ground_truth_csv: str = OUTPUT_FILE_ANNOTATION_TARGET_LIST,
    features_csv: str = OUTPUT_FILE_FEATURES_FOR_EVALUATION,
    llm_predictions_csv: str = OUTPUT_FILE_PREDICTION_LLM,
    output_file_name: str = 'evaluation_metrics_summary.csv'
):
    """
    結果分析のメイン処理を実行します。
    """
    df_eval_base = load_and_merge_evaluation_data(ground_truth_csv, features_csv, llm_predictions_csv)
    if df_eval_base.empty:
        return

    df_eval_with_hybrid = generate_hybrid_predictions(df_eval_base)

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
    save_evaluation_results(df_metrics, output_file_name)

if __name__ == "__main__":
    main_analyze_results()
