# Data Usage Validator

## 概要
このプロジェクトは、学術論文におけるデータ利用を検証するためのフレームワークを提供します。大規模言語モデル（LLM）とルールベースのアプローチを組み合わせて、論文中のデータ利用箇所を特定し、その妥当性を評価します。

**主な目的:**
*   科学論文が特定のデータ論文のデータを「実際に使用しているか」を自動的に判定する手法を開発し、その性能を評価する。
*   LLMを活用したデータ利用検証の有効性を探り、ルールベースの手法や人間によるレビューと組み合わせることで、堅牢で効率的な検証パイプラインを構築する。

## 特徴
- **モジュール化されたパイプライン**: データ収集からLLM検証、評価、レビューまでの一連のプロセスが、独立したPythonスクリプトとして `pipeline/` ディレクトリに整理されています。
- **コマンドラインからの実行**: `run_pipeline.py` スクリプトを通じて、各フェーズを個別に、または一括で実行・制御できます。
- **LLMベースの検証**: 事前定義されたプロンプト（few-shot CoT, zero-shot）を用いて、LLMが論文中のデータ利用を評価します。
- **ルールベースの検証**: 特定のルールに基づいてデータ利用を検証します（`src/xml_processor.py` に関連ロジック）。
- **結果分析**: 評価指標（F1スコア、混同行列など）を用いて、検証結果を分析し、可視化します。
- **モックテスト**: APIリクエストなしでロジックを検証できる単体テストが `tests/` ディレクトリに用意されています。
- **詳細ドキュメント**: `docs/` ディレクトリに、パイプラインの利用方法や実験の詳細に関するドキュメントが提供されています。

## プロジェクト構造

- `data/`:
    - `ground_truth/`: アノテーションされたデータ利用の正解データ。
    - `processed/`: 収集および処理された論文データ。
    - `raw/fulltext/`: ダウンロードされた全文XMLファイル。
- `docs/`:
    - パイプラインの利用方法 (`pipeline_usage.md`) や実験の詳細 (`experiment_details.md`) に関するドキュメント。
- `notebook/`:
    - データ収集、前処理、LLMによる検証、結果分析など、プロジェクトの各段階を実行するためのJupyter Notebook（主に開発・探索用）。
- `pipeline/`:
    - プロジェクトの主要なワークフローを構成するモジュール化されたPythonスクリプト。
    - `main_pipeline.py`: 全てのパイプラインフェーズを統合し、コマンドライン引数で制御するメインスクリプト。
- `prompts/`:
    - LLMにデータ利用検証タスクを指示するためのプロンプトファイル。
- `results/`:
    - `figures/`: 評価結果のグラフや図（例: 混同行列、F1スコア比較）。
    - `tables/`: 評価指標や詳細な結果データ。
- `src/`:
    - `config.py`: APIキーやファイルパスなどの設定情報。
    - `scopus_api.py`: Scopus APIとの連携に関する関数群。
    - `collect_data.py`: データ論文収集のロジック。
    - `collect_citing_papers.py`: 引用論文の収集と全文XMLダウンロードのロジック。
    - `sampling.py`: アノテーション用サンプリングリスト作成のロジック。
    - `xml_processor.py`: XML解析とルールベースの特徴量抽出のロジック。
    - `text_extractor.py`: XMLからのアブストラクト・全文テキスト抽出のロジック。
    - `data_processor.py`: テキスト抽出のロジック。
    - `llm_validator.py`: LLMによるデータ利用判定のロジック。
    - `evaluation.py`: 評価結果の分析と指標計算のロジック。
    - `review_and_correction.py`: LLM予測結果のレビューと正解データ修正のロジック。
- `tests/`:
    - 各パイプラインモジュールおよび主要な `src` モジュールに対応する単体テスト。
- `run_pipeline.py`: `pipeline/main_pipeline.py` を実行するためのラッパースクリプト。
- `requirements.txt`: プロジェクトの依存関係リスト。

## はじめに

### 必要なもの
- Python 3.x
- Scopus APIキー
- Gemini APIキー (LLMベースの検証を実行する場合)

### セットアップ
1. このリポジトリをクローンします。
   ```bash
   git clone https://github.com/ku6kaw/DataUsageValidator.git
   cd DataUsageValidator
   ```
2. 必要なPythonライブラリをインストールします。
   ```bash
   pip install -r requirements.txt
   ```
3. `src/config.py` を開き、`SCOPUS_API_KEY` と `GEMINI_API_KEY` をご自身のAPIキーに置き換えてください。

### 実行方法
パイプラインは `run_pipeline.py` スクリプトを通じて実行されます。詳細な実行方法やオプションについては、`docs/pipeline_usage.md` を参照してください。

*   **全てのパイプラインを実行**:
    ```bash
    python run_pipeline.py --run_all
    ```
*   **利用可能なオプションの確認**:
    ```bash
    python run_pipeline.py --help
    ```

### テストの実行
全ての単体テストを実行するには、以下のコマンドを使用します。
```bash
python -m unittest discover tests
```

## 詳細ドキュメント

*   **パイプライン利用ガイド**: `docs/pipeline_usage.md`
*   **実験詳細**: `docs/experiment_details.md`


