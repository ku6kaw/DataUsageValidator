import argparse
import os
from pipeline.collect_data_pipeline import run_collect_data_pipeline
from pipeline.collect_citing_papers_pipeline import run_collect_citing_papers_pipeline
from pipeline.prepare_data_pipeline import run_prepare_data_pipeline
from pipeline.llm_validation_pipeline import run_llm_validation_pipeline
from pipeline.evaluate_results_pipeline import run_evaluate_results_pipeline
from pipeline.review_and_correct_pipeline import run_review_and_correction_pipeline
from src.config import (
    SCOPUS_API_KEY, GEMINI_API_KEY, GEMINI_MODEL_NAME,
    OUTPUT_FILE_DATA_PAPERS, OUTPUT_FILE_CITING_PAPERS_WITH_PATHS,
    OUTPUT_FILE_ANNOTATION_TARGET_LIST, OUTPUT_FILE_SAMPLES_WITH_TEXT,
    OUTPUT_FILE_PREDICTION_LLM, OUTPUT_FILE_FEATURES_FOR_EVALUATION,
    OUTPUT_DIR_PROCESSED, XML_OUTPUT_DIR
)

def main_pipeline(args):
    """
    データ利用検証の全パイプラインを実行します。
    """
    print("========================================")
    print("=== データ利用検証パイプライン開始 ===")
    print("========================================")

    # フェーズ1: データ論文収集
    if args.run_collect_data:
        run_collect_data_pipeline(
            api_key=args.scopus_api_key,
            output_file=OUTPUT_FILE_DATA_PAPERS,
            output_dir=OUTPUT_DIR_PROCESSED
        )

    # フェーズ2: 引用論文収集
    if args.run_collect_citing_papers:
        run_collect_citing_papers_pipeline(
            api_key=args.scopus_api_key,
            input_data_papers_csv=OUTPUT_FILE_DATA_PAPERS,
            output_citing_papers_csv=OUTPUT_FILE_CITING_PAPERS_WITH_PATHS,
            xml_output_dir=XML_OUTPUT_DIR,
            min_citations=args.min_citations,
            max_workers_download_xml=args.max_workers_download_xml,
            retry_failed=args.retry_failed_downloads
        )

    # フェーズ3: データ準備とサンプリング、XML処理、テキスト抽出
    if args.run_prepare_data:
        run_prepare_data_pipeline(
            citing_papers_master_csv=OUTPUT_FILE_CITING_PAPERS_WITH_PATHS,
            annotation_target_list_csv=OUTPUT_FILE_ANNOTATION_TARGET_LIST,
            samples_with_text_csv=OUTPUT_FILE_SAMPLES_WITH_TEXT,
            processed_output_dir=OUTPUT_DIR_PROCESSED,
            sample_size=args.sample_size,
            random_state=args.random_state
        )

    # フェーズ4: LLM検証
    if args.run_llm_validation:
        run_llm_validation_pipeline(
            api_key=args.gemini_api_key,
            model_name=args.gemini_model_name,
            input_samples_csv=OUTPUT_FILE_SAMPLES_WITH_TEXT,
            output_predictions_csv=OUTPUT_FILE_PREDICTION_LLM,
            run_abstract_prediction=args.run_abstract_prediction,
            run_fulltext_zeroshot_prediction=args.run_fulltext_zeroshot_prediction,
            run_fulltext_fewshot_cot_prediction=args.run_fulltext_fewshot_cot_prediction,
            retry_failed_abstract=args.retry_failed_abstract,
            retry_failed_fulltext_zeroshot=args.retry_failed_fulltext_zeroshot,
            retry_failed_fulltext_fewshot_cot=args.retry_failed_fulltext_fewshot_cot,
            sleep_time=args.llm_sleep_time,
            timeout=args.llm_timeout
        )

    # フェーズ5: 評価と分析
    if args.run_evaluate_results:
        run_evaluate_results_pipeline(
            ground_truth_csv=OUTPUT_FILE_ANNOTATION_TARGET_LIST,
            features_csv=OUTPUT_FILE_FEATURES_FOR_EVALUATION, # このファイルは別途生成される必要がある
            llm_predictions_csv=OUTPUT_FILE_PREDICTION_LLM,
            output_metrics_file_name='evaluation_metrics_summary.csv'
        )

    # フェーズ6: レビューと修正
    if args.run_review_and_correct:
        run_review_and_correct_pipeline(
            ground_truth_csv=OUTPUT_FILE_ANNOTATION_TARGET_LIST,
            llm_predictions_csv=OUTPUT_FILE_PREDICTION_LLM,
            samples_with_text_csv=OUTPUT_FILE_SAMPLES_WITH_TEXT,
            best_model_column=args.best_model_column,
            corrections=None, # コマンドライン引数からは直接修正を渡さない
            generate_prompts=True
        )

    print("========================================")
    print("=== データ利用検証パイプライン完了 ===")
    print("========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="データ利用検証パイプラインを実行します。")

    # グローバル設定
    parser.add_argument("--scopus_api_key", type=str, default=SCOPUS_API_KEY, help="Scopus APIキー")
    parser.add_argument("--gemini_api_key", type=str, default=GEMINI_API_KEY, help="Gemini APIキー")
    parser.add_argument("--gemini_model_name", type=str, default=GEMINI_MODEL_NAME, help="使用するGeminiモデル名")
    parser.add_argument("--random_state", type=int, default=42, help="ランダムシード")

    # 各フェーズの実行フラグ
    parser.add_argument("--run_collect_data", action="store_true", help="データ論文収集フェーズを実行")
    parser.add_argument("--run_collect_citing_papers", action="store_true", help="引用論文収集フェーズを実行")
    parser.add_argument("--run_prepare_data", action="store_true", help="データ準備フェーズを実行")
    parser.add_argument("--run_llm_validation", action="store_true", help="LLM検証フェーズを実行")
    parser.add_argument("--run_evaluate_results", action="store_true", help="評価と分析フェーズを実行")
    parser.add_argument("--run_review_and_correct", action="store_true", help="レビューと修正フェーズを実行")
    
    # 全てのフェーズを実行するショートカット
    parser.add_argument("--run_all", action="store_true", help="全てのパイプラインフェーズを実行")

    # collect_citing_papers_pipeline の引数
    parser.add_argument("--min_citations", type=int, default=10, help="引用論文収集対象とするデータ論文の最小被引用数")
    parser.add_argument("--max_workers_download_xml", type=int, default=10, help="XMLダウンロード時の並列処理スレッド数")
    parser.add_argument("--retry_failed_downloads", action="store_true", help="失敗したXMLダウンロードを再試行")

    # prepare_data_pipeline の引数
    parser.add_argument("--sample_size", type=int, default=200, help="アノテーション用サンプル数")

    # llm_validation_pipeline の引数
    parser.add_argument("--run_abstract_prediction", action="store_true", help="アブストラクトを用いたLLM予測を実行")
    parser.add_argument("--run_fulltext_zeroshot_prediction", action="store_true", help="全文を用いたZero-shot LLM予測を実行")
    parser.add_argument("--run_fulltext_fewshot_cot_prediction", action="store_true", help="全文を用いたFew-shot CoT LLM予測を実行")
    parser.add_argument("--retry_failed_abstract", action="store_true", help="失敗したアブストラクト予測を再試行")
    parser.add_argument("--retry_failed_fulltext_zeroshot", action="store_true", help="失敗した全文Zero-shot予測を再試行")
    parser.add_argument("--retry_failed_fulltext_fewshot_cot", action="store_true", help="失敗した全文Few-shot CoT予測を再試行")
    parser.add_argument("--llm_sleep_time", type=float, default=1.0, help="LLM APIリクエスト間の待機時間（秒）")
    parser.add_argument("--llm_timeout", type=int, default=180, help="LLM APIリクエストのタイムアウト時間（秒）")

    # review_and_correct_pipeline の引数
    parser.add_argument("--best_model_column", type=str, default='prediction_rule3_gemini-2_5-flash', help="レビュー対象のLLM予測結果カラム名")

    args = parser.parse_args()

    if args.run_all:
        args.run_collect_data = True
        args.run_collect_citing_papers = True
        args.run_prepare_data = True
        args.run_llm_validation = True
        args.run_evaluate_results = True
        args.run_review_and_correct = True
        args.run_abstract_prediction = True
        args.run_fulltext_zeroshot_prediction = True
        # args.run_fulltext_fewshot_cot_prediction = True # Few-shot CoTはデフォルトでは実行しない

    main_pipeline(args)
