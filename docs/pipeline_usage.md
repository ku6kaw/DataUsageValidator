# データ利用検証パイプライン利用ガイド

このドキュメントは、科学論文におけるデータ利用の検証と分析を行うためのパイプラインの利用方法を説明します。このパイプラインは、複数のフェーズに分かれており、それぞれが独立したPythonスクリプトとしてモジュール化されています。

## 1. はじめに

このパイプラインは、以下の主要なフェーズで構成されています。

1.  **データ論文収集**: Scopus APIを使用してデータ論文を収集します。
2.  **引用論文収集**: 収集したデータ論文を引用している論文のメタデータと全文XMLを収集します。
3.  **データ準備**: 引用論文からアノテーション対象のサンプルリストを作成し、XMLからテキストを抽出します。
4.  **LLM検証**: 大規模言語モデル (LLM) を使用して、論文のデータ利用を自動的に予測します。
5.  **評価と分析**: LLMの予測結果を正解データと比較し、様々な評価指標を計算・分析します。
6.  **レビューと修正**: LLMの予測と人間の判断の食い違いを特定し、必要に応じて正解データを修正します。

## 2. 環境設定

### APIキーの設定

`src/config.py` ファイルを開き、以下のAPIキーを各自のものに置き換えてください。

*   `SCOPUS_API_KEY`: Scopus APIのキー
*   `GEMINI_API_KEY`: Google Gemini APIのキー

```python
# src/config.py
SCOPUS_API_KEY = "YOUR_API_KEY" # Replace with your actual API key
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY" # Replace with your actual Gemini API key
```

### 依存関係のインストール

プロジェクトの依存関係は `requirements.txt` に記述されています。以下のコマンドでインストールしてください。

```bash
pip install -r requirements.txt
```

## 3. パイプラインの実行

パイプラインは `run_pipeline.py` スクリプトを通じて実行されます。各フェーズはコマンドライン引数で有効/無効を切り替えることができます。

### 全てのフェーズを実行する

```bash
python run_pipeline.py --run_all
```

### 特定のフェーズのみを実行する

各フェーズは `--run_collect_data`, `--run_collect_citing_papers`, `--run_prepare_data`, `--run_llm_validation`, `--run_evaluate_results`, `--run_review_and_correct` のフラグで個別に実行できます。

例: データ論文収集と引用論文収集のみを実行する場合

```bash
python run_pipeline.py --run_collect_data --run_collect_citing_papers
```

### 各フェーズのオプション引数

`run_pipeline.py` は、各フェーズの動作をカスタマイズするための多くの引数を提供します。`--help` オプションで詳細を確認できます。

```bash
python run_pipeline.py --help
```

以下に主要な引数の一部を示します。

*   `--scopus_api_key`: Scopus APIキー (config.pyのデフォルトを上書き)
*   `--gemini_api_key`: Gemini APIキー (config.pyのデフォルトを上書き)
*   `--gemini_model_name`: 使用するGeminiモデル名 (デフォルト: `gemini-1.5-flash`)
*   `--random_state`: ランダムサンプリングのシード (デフォルト: `42`)
*   `--min_citations`: 引用論文収集対象とするデータ論文の最小被引用数 (デフォルト: `10`)
*   `--max_workers_download_xml`: XMLダウンロード時の並列処理スレッド数 (デフォルト: `10`)
*   `--retry_failed_downloads`: 失敗したXMLダウンロードを再試行するかどうか
*   `--sample_size`: アノテーション用サンプル数 (デフォルト: `200`)
*   `--run_abstract_prediction`: アブストラクトを用いたLLM予測を実行するかどうか
*   `--run_fulltext_zeroshot_prediction`: 全文を用いたZero-shot LLM予測を実行するかどうか
*   `--run_fulltext_fewshot_cot_prediction`: 全文を用いたFew-shot CoT LLM予測を実行するかどうか (デフォルト: `False`)
*   `--retry_failed_abstract`: 失敗したアブストラクト予測を再試行するかどうか
*   `--retry_failed_fulltext_zeroshot`: 失敗した全文Zero-shot予測を再試行するかどうか
*   `--retry_failed_fulltext_fewshot_cot`: 失敗した全文Few-shot CoT予測を再試行するかどうか
*   `--llm_sleep_time`: LLM APIリクエスト間の待機時間（秒） (デフォルト: `1.0`)
*   `--llm_timeout`: LLM APIリクエストのタイムアウト時間（秒） (デフォルト: `180`)
*   `--best_model_column`: レビュー対象のLLM予測結果カラム名 (デフォルト: `prediction_rule3_gemini-2_5-flash`)

## 4. テストの実行

各パイプラインモジュールには対応する単体テストが `tests/` ディレクトリに用意されています。APIリクエストを伴わないモックデータを使用したテストです。

全てのテストを実行するには、以下のコマンドを使用します。

```bash
python -m unittest discover tests
```

## 5. 出力ファイル

各フェーズの出力ファイルは `data/processed/` および `results/tables/` ディレクトリに保存されます。

*   `data/processed/data_papers.csv`: 収集されたデータ論文のリスト
*   `data/processed/citing_papers_with_paths.csv`: 収集された引用論文のリストとXMLパス
*   `data/ground_truth/annotation_target_list.csv`: アノテーション対象のサンプルリスト
*   `data/processed/samples_with_text.csv`: 抽出されたテキストを含むサンプルデータ
*   `data/processed/prediction_llm.csv`: LLMによる予測結果
*   `results/tables/evaluation_metrics_summary.csv`: 評価指標のサマリー

## 6. 実験の再現性

このパイプラインは、以下の点に配慮して実験の再現性を高めるように設計されています。

*   **モジュール化**: 各フェーズが独立したスクリプトとして提供され、個別に実行・検証が可能です。
*   **設定の一元化**: `src/config.py` でAPIキー、ファイルパス、モデル名などの設定を一元管理します。
*   **コマンドライン引数**: 実行時に主要なパラメータを調整できるため、異なる設定での実験が容易です。
*   **モックテスト**: APIリクエストなしでロジックを検証できるテストが提供され、開発とデバッグを支援します。
*   **ランダムシード**: サンプリングなどのランダムな処理には `random_state` を設定することで、結果の再現性を保証します。

## 7. 補足

*   `src/evaluation.py` で使用される `OUTPUT_FILE_FEATURES_FOR_EVALUATION` (ルールベースの特徴量CSV) は、このパイプラインでは直接生成されません。これは、別途ルールベースの検証プロセスが存在し、その結果がこのファイルに保存されることを想定しています。必要に応じて、このファイルを作成するフェーズを追加するか、既存のノートブックから生成してください。
*   `src/review_and_correction.py` の `generate_review_prompts` 関数は、LLMに再確認を促すプロンプトを標準出力に表示します。これをコピーしてLLMに与え、その応答を手動で `corrections` 辞書に反映させることで、正解データを更新できます。
