import subprocess
import sys
import os

def run_pipeline_script():
    """
    main_pipeline.py を実行するためのラッパースクリプト。
    コマンドライン引数を透過的に渡します。
    """
    # main_pipeline.py のパスを構築
    pipeline_script_path = os.path.join('pipeline', 'main_pipeline.py')

    # main_pipeline.py に渡す引数を構築
    # 最初の引数 (スクリプト名) を除外
    args_to_pass = sys.argv[1:]

    command = [sys.executable, pipeline_script_path] + args_to_pass
    
    print(f"実行コマンド: {' '.join(command)}")
    
    try:
        # サブプロセスとして実行し、出力をリアルタイムで表示
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        for line in process.stdout:
            print(line, end='')
        process.wait()

        if process.returncode != 0:
            print(f"エラー: パイプラインの実行がコード {process.returncode} で終了しました。")
            sys.exit(process.returncode)
        else:
            print("パイプラインの実行が正常に完了しました。")

    except FileNotFoundError:
        print(f"エラー: '{pipeline_script_path}' が見つかりません。パスを確認してください。")
        sys.exit(1)
    except Exception as e:
        print(f"パイプラインの実行中に予期せぬエラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline_script()
