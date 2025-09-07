import pandas as pd
import os

from src.config import (
    OUTPUT_FILE_ANNOTATION_TARGET_LIST,
    OUTPUT_FILE_PREDICTION_LLM,
    OUTPUT_FILE_SAMPLES_WITH_TEXT
)

def load_and_merge_review_data(
    ground_truth_csv: str = OUTPUT_FILE_ANNOTATION_TARGET_LIST,
    llm_predictions_csv: str = OUTPUT_FILE_PREDICTION_LLM,
    samples_with_text_csv: str = OUTPUT_FILE_SAMPLES_WITH_TEXT,
    best_model_column: str = 'prediction_rule3_gemini-2_5-flash'
) -> pd.DataFrame:
    """
    正解データ、LLM予測結果、テキスト抽出済みデータを読み込み、結合します。

    Args:
        ground_truth_csv (str): 正解データCSVファイルのパス。
        llm_predictions_csv (str): LLM予測結果CSVファイルのパス。
        samples_with_text_csv (str): テキスト抽出済みデータCSVファイルのパス。
        best_model_column (str): 評価対象のLLM予測結果カラム名。

    Returns:
        pd.DataFrame: 結合されたレビュー用DataFrame。
    """
    try:
        df_gt = pd.read_csv(ground_truth_csv)
        df_llm = pd.read_csv(llm_predictions_csv)
        df_text = pd.read_csv(samples_with_text_csv)

        df_merged = pd.merge(df_gt, df_llm, on='citing_paper_doi', how='inner', suffixes=('_gt', ''))
        df_review = pd.merge(df_merged, df_text, on='citing_paper_doi', how='inner', suffixes=('', '_text'))
        
        df_review.dropna(subset=['is_data_used_gt', best_model_column], inplace=True)
        df_review = df_review[df_review[best_model_column] != -1]
        df_review[best_model_column] = df_review[best_model_column].astype(int)
        df_review['is_data_used_gt'] = df_review['is_data_used_gt'].astype(int)
        
        print("✅ 正解データとLLM予測結果の読み込み・結合が完了しました。")
        return df_review

    except FileNotFoundError as e:
        print(f"❌ エラー: ファイルが見つかりません。パスを確認してください。 {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"データの読み込みまたは結合中にエラーが発生しました: {e}")
        return pd.DataFrame()

def identify_disagreements(df_review: pd.DataFrame, best_model_column: str) -> pd.DataFrame:
    """
    人間の判断とLLMの判断が食い違った論文を特定します。

    Args:
        df_review (pd.DataFrame): レビュー用DataFrame。
        best_model_column (str): 評価対象のLLM予測結果カラム名。

    Returns:
        pd.DataFrame: 判断が食い違った論文のDataFrame。
    """
    if df_review.empty:
        return pd.DataFrame()

    disagreements = df_review[df_review['is_data_used_gt'] != df_review[best_model_column]].copy()
    disagreements['Human_Label'] = disagreements['is_data_used_gt'].map({1: 'Used', 0: 'Not Used'})
    disagreements['LLM_Prediction'] = disagreements[best_model_column].map({1: 'Used', 0: 'Not Used'})

    print("\n" + "="*70)
    print("【人間の判断とLLMの判断が食い違った「怪しい」論文リスト】")
    print("="*70)
    
    if not disagreements.empty:
        print(f"合計 {len(disagreements)} 件の食い違いが見つかりました。")
        print(disagreements[['citing_paper_doi', 'citing_paper_title', 'Human_Label', 'LLM_Prediction']].to_string())
    else:
        print("✅ 人間の判断とLLMの判断はすべて一致していました。")
        
    return disagreements

def generate_review_prompts(disagreements: pd.DataFrame, best_model_column: str):
    """
    判断が食い違った論文について、LLMに再確認を促すプロンプトを生成します。

    Args:
        disagreements (pd.DataFrame): 判断が食い違った論文のDataFrame。
        best_model_column (str): 評価対象のLLM予測結果カラム名。
    """
    if not disagreements.empty:
        print("\n" + "="*70)
        print(f"【確認が必要な {len(disagreements)} 件の論文について、以下のプロンプトをコピーして私に送信してください】")
        print("="*70)
        
        for index, row in disagreements.iterrows():
            human_label = "Used" if row['is_data_used_gt'] == 1 else "Not Used"
            llm_label = "Used" if row[best_model_column] == 1 else "Not Used"
            
            prompt_template = f"""
私は、ある論文が指定されたデータ論文のデータを「実際に使用しているか」を手動でラベル付けしました。
しかし、作成したAIモデルの判断と私の判断が食い違ったため、第三者の意見を参考に、私の判断が正しかったかを再確認したいです。

以下の【入力データ】と【判定基準】を基に、この論文はデータを「使用している」と判断すべきか、「使用していない」と判断すべきか、あなたの専門的な見解を教えてください。

---
### 【判定基準】
* **"Used":** 論文の主張や結論（性能評価、分析結果、比較など）を導き出すために、データセットが分析、訓練、評価、性能比較などのプロセスに直接的に用いられている状態。
* **"Not Used":** 研究の背景説明、関連研究の紹介、あるいは今後の展望としてデータセット名に言及しているだけの状態。

---
### 【入力データ】

**引用元データ論文のタイトル:**
{row['cited_data_paper_title']}

**被引用論文のタイトル:**
{row['citing_paper_title']}

**被引用論文の全文テキスト:**
{row['full_text']}

---
### 【状況】
* **私の最初の判断:** {human_label}
* **AIモデルの判断:** {llm_label}

---
### 【あなたのタスク】
上記の情報を踏まえ、最終的にどちらの判断がより妥当と考えられるか、その根拠となるテキスト中の箇所を引用しながら、あなたの分析結果を提示してください。
"""
            print("\n\n" + "#"*20 + f"  確認用プロンプト {index+1} / DOI: {row['citing_paper_doi']}  " + "#"*20)
            print(prompt_template)
            print("#"*70 + "\n")
    else:
        print("✅ 人間の判断とLLMの判断はすべて一致していました。確認作業は不要です。")

def apply_corrections_to_ground_truth(
    corrections: dict, 
    ground_truth_csv: str = OUTPUT_FILE_ANNOTATION_TARGET_LIST
):
    """
    正解データCSVファイルに手動修正を適用します。

    Args:
        corrections (dict): DOIをキー、新しいラベル（0または1）を値とする辞書。
        ground_truth_csv (str): 正解データCSVファイルのパス。
    """
    if not corrections:
        print("修正対象が指定されていません。")
        return

    try:
        df_gt = pd.read_csv(ground_truth_csv)
        print(f"元の '{os.path.basename(ground_truth_csv)}' を読み込みました。")

        updated_count = 0
        for doi, new_label in corrections.items():
            target_indices = df_gt[df_gt['citing_paper_doi'] == doi].index
            
            if not target_indices.empty:
                df_gt.loc[target_indices, 'is_data_used_gt'] = new_label
                print(f"  - DOI: {doi} のラベルを {new_label} に更新しました。")
                updated_count += 1
            else:
                print(f"  - 警告: DOI {doi} は見つかりませんでした。")

        if updated_count > 0:
            df_gt.to_csv(ground_truth_csv, index=False, encoding='utf-8-sig')
            print(f"\n✅ {updated_count}件の修正をファイルに保存しました。")

    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")

def main_review_and_correction(
    best_model_column: str = 'prediction_rule3_gemini-2_5-flash',
    corrections: dict = None
):
    """
    レビューと修正のメイン処理を実行します。
    """
    df_review = load_and_merge_review_data(best_model_column=best_model_column)
    if df_review.empty:
        return

    disagreements = identify_disagreements(df_review, best_model_column)
    generate_review_prompts(disagreements)

    if corrections:
        apply_corrections_to_ground_truth(corrections)

if __name__ == "__main__":
    # 例として、レビューとプロンプト生成を実行
    main_review_and_correction(best_model_column='prediction_rule3_gemini-2_5-flash')

    # 手動修正を適用する例 (必要に応じてコメントを外して実行)
    # corrections_to_apply = {
    #     "10.1016/j.jprot.2022.104578": 1, # このDOIのラベルを1に修正
    #     "10.1016/j.ccell.2021.04.015": 0  # このDOIのラベルを0に修正
    # }
    # main_review_and_correction(corrections=corrections_to_apply)
