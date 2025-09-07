import pandas as pd
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

from src.config import (
    OUTPUT_FILE_CITING_PAPERS_WITH_PATHS,
    OUTPUT_DIR_PROCESSED,
    OUTPUT_FILE_ANNOTATION_TARGET_LIST,
    OUTPUT_FILE_SAMPLES_WITH_TEXT
)
from src.text_extractor import extract_abstract_robustly, extract_full_text_robustly

def extract_text_from_xml_files(
    annotation_list_csv: str = OUTPUT_FILE_ANNOTATION_TARGET_LIST,
    master_list_csv: str = OUTPUT_FILE_CITING_PAPERS_WITH_PATHS,
    output_dir: str = OUTPUT_DIR_PROCESSED,
    output_file_name: str = OUTPUT_FILE_SAMPLES_WITH_TEXT
) -> pd.DataFrame:
    """
    アノテーション対象の論文リストとマスターリストをマージし、XMLからアブストラクトと全文を抽出します。

    Args:
        annotation_list_csv (str): アノテーション対象の論文リストCSVパス。
        master_list_csv (str): 全ての引用論文のマスターリストCSVパス（XMLパス情報を含む）。
        output_dir (str): 結果を保存するディレクトリ。
        output_file_name (str): 結果を保存するファイル名。

    Returns:
        pd.DataFrame: 抽出されたテキストが追加されたDataFrame。
    """
    output_file_path = os.path.join(output_dir, output_file_name)

    try:
        df_targets = pd.read_csv(annotation_list_csv)
        df_targets.drop_duplicates(subset=['citing_paper_doi'], inplace=True)
        df_master = pd.read_csv(master_list_csv)
        df_master_subset = df_master[['citing_paper_doi', 'fulltext_xml_path']].drop_duplicates(subset=['citing_paper_doi'])
        df_to_process = pd.merge(df_targets, df_master_subset, on='citing_paper_doi', how='left')
        
        extracted_data_list = []
        print(f"処理対象の {len(df_to_process)} 件の論文からテキストを抽出します...")
        
        for index, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="テキスト抽出中"):
            xml_path = row.get('fulltext_xml_path')
            abstract = None
            full_text = None
            
            if pd.notna(xml_path) and os.path.exists(xml_path):
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    abstract = extract_abstract_robustly(root)
                    full_text = extract_full_text_robustly(root)
                except ET.ParseError:
                    print(f"警告: DOI {row['citing_paper_doi']} のXMLをパースできませんでした。")

            extracted_data_list.append({
                'citing_paper_eid': row['citing_paper_eid'],
                'citing_paper_doi': row['citing_paper_doi'],
                'citing_paper_title': row['citing_paper_title'],
                'cited_data_paper_title': row['cited_data_paper_title'],
                'abstract': abstract,
                'full_text': full_text
            })

        df_final = pd.DataFrame(extracted_data_list)
        os.makedirs(output_dir, exist_ok=True)
        df_final.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        
        print(f"\n処理完了。")
        print(f"抽出したテキストを '{output_file_path}' に保存しました。")
        print("\n--- 監査を実行します ---")
        print(df_final.info())
        return df_final

    except FileNotFoundError as e:
        print(f"必要なファイルが見つかりません: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"メイン処理中にエラーが発生しました: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    extract_text_from_xml_files()
