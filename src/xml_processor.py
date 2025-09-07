import xml.etree.ElementTree as ET
import pandas as pd
import os
from tqdm import tqdm

# XMLの名前空間
namespaces = {'ce': 'http://www.elsevier.com/xml/common/dtd', 'sb': 'http://www.elsevier.com/xml/common/struct-bib/dtd', 'ja': 'http://www.elsevier.com/xml/ja/dtd'}

def get_citation_map_et(root_element: ET.Element) -> dict:
    """
    参考文献リストから {ref_id: '文献情報'} の辞書を作成する。

    Args:
        root_element (ET.Element): XMLのルート要素。

    Returns:
        dict: 参考文献IDをキー、文献情報を値とする辞書。
    """
    citation_map = {}
    references = root_element.findall('.//ce:bibliography/ce:bibliography-sec/ce:bib-reference', namespaces)
    for ref in references:
        ref_id = ref.get('id')
        source_text_element = ref.find('.//ce:source-text', namespaces)
        citation_text = source_text_element.text if source_text_element is not None else ''.join(ref.itertext())
        if ref_id:
            citation_map[ref_id] = citation_text.strip() if citation_text else 'N/A'
    return citation_map

def find_target_ref_id(citation_map: dict, target_title: str) -> str or None:
    """
    参考文献マップとデータ論文タイトルから、対応するRef IDを見つける。

    Args:
        citation_map (dict): 参考文献IDと文献情報の辞書。
        target_title (str): 検索対象のデータ論文タイトル。

    Returns:
        str or None: 見つかったRef ID、または見つからなかった場合はNone。
    """
    for ref_id, full_citation in citation_map.items():
        if target_title.lower() in full_citation.lower():
            return ref_id
    return None

def parse_sections_recursive(element: ET.Element) -> list:
    """
    XML要素から、全セクションのタイトルと、各セクション内の引用IDリストを抽出する（再帰的）。

    Args:
        element (ET.Element): 解析対象のXML要素。

    Returns:
        list: 各セクションのタイトルと引用IDリストを含む辞書のリスト。
    """
    sections_data = []
    for section in element.findall('./ce:section', namespaces):
        title_tag = section.find('./ce:section-title', namespaces)
        sec_title = title_tag.text.strip() if title_tag is not None and title_tag.text else 'No Title'
        
        citations_in_section = []
        paragraphs = section.findall('./ce:para', namespaces)
        for p in paragraphs:
            cross_refs = p.findall('.//ce:cross-ref', namespaces)
            for xref in cross_refs:
                if xref.get('refid'):
                    ref_ids = xref.get('refid').split()
                    for ref_id in ref_ids:
                        citations_in_section.append(ref_id)
                        
        sections_data.append({'title': sec_title, 'citations': citations_in_section})
        sections_data.extend(parse_sections_recursive(section)) # 再帰呼び出し
    return sections_data

def analyze_single_xml(xml_path: str, target_data_paper_title: str) -> tuple:
    """
    1つのXMLファイルを解析し、特徴量と判定結果を抽出するメイン関数。

    Args:
        xml_path (str): 解析対象のXMLファイルのパス。
        target_data_paper_title (str): 引用されているデータ論文のタイトル。

    Returns:
        tuple: (mention_count, mentioned_sections, prediction_rule1, prediction_rule2)
               エラーの場合は (-1, ['parsing_error'], -1, -1)。
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        citation_map = get_citation_map_et(root)
        target_ref_id = find_target_ref_id(citation_map, target_data_paper_title)
        if not target_ref_id:
            return 0, [], 0, 0

        top_level_sections = root.find('.//ja:body/ce:sections', namespaces)
        if not top_level_sections:
            return 0, [], 0, 0
        
        all_sections_data = parse_sections_recursive(top_level_sections)
        
        mention_count = 0
        mentioned_sections = []
        keywords_to_check = ['data', 'method', 'experiment']
        contains_keyword = False
        
        for section in all_sections_data:
            count_in_section = section['citations'].count(target_ref_id)
            if count_in_section > 0:
                mention_count += count_in_section
                section_title = section['title']
                mentioned_sections.append(section_title)
                if any(keyword in section_title.lower() for keyword in keywords_to_check):
                    contains_keyword = True
        
        prediction_rule1 = 1 if mention_count >= 2 else 0
        prediction_rule2 = 1 if contains_keyword else 0
        
        return mention_count, list(set(mentioned_sections)), prediction_rule1, prediction_rule2

    except Exception:
        return -1, ['parsing_error'], -1, -1

def process_xml_for_features(
    df_targets: pd.DataFrame, 
    df_master: pd.DataFrame, 
    output_dir: str, 
    output_file_name: str
) -> pd.DataFrame:
    """
    アノテーション対象の論文リストとマスターリストをマージし、XMLを解析して特徴量を抽出します。

    Args:
        df_targets (pd.DataFrame): アノテーション対象の論文リスト。
        df_master (pd.DataFrame): 全ての引用論文のマスターリスト（パス情報を含む）。
        output_dir (str): 結果を保存するディレクトリ。
        output_file_name (str): 結果を保存するファイル名。

    Returns:
        pd.DataFrame: 特徴量が追加されたDataFrame。
    """
    output_file_path = os.path.join(output_dir, output_file_name)

    merge_keys = ['citing_paper_eid', 'citing_paper_doi', 'citing_paper_title', 'cited_data_paper_title']
    df_to_process = pd.merge(df_targets, df_master.drop_duplicates(subset=merge_keys), on=merge_keys, how='left')
    
    results_list = []
    print(f"アノテーション対象 {len(df_to_process)} 件のXMLを解析します...")
    
    for index, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="特徴量抽出中"):
        xml_path = row['fulltext_xml_path']
        target_title = row['cited_data_paper_title']
        
        if pd.notna(xml_path) and os.path.exists(xml_path):
            count, sections, pred1, pred2 = analyze_single_xml(xml_path, target_title)
        else:
            count, sections, pred1, pred2 = -1, ['file_not_found'], -1, -1

        result_row = row.to_dict()
        result_row['mention_count'] = count
        result_row['mentioned_sections'] = sections
        result_row['prediction_rule1'] = pred1
        result_row['prediction_rule2'] = pred2
        results_list.append(result_row)

    df_final = pd.DataFrame(results_list)
    
    columns_to_save = [
        'citing_paper_eid', 
        'citing_paper_doi', 
        'citing_paper_title', 
        'cited_data_paper_title',
        'mention_count',
        'mentioned_sections',
        'prediction_rule1',
        'prediction_rule2'
    ]
    df_to_save = df_final[columns_to_save]
    
    os.makedirs(output_dir, exist_ok=True)
    df_to_save.to_csv(output_file_path, index=False, encoding='utf-8-sig')

    print(f"\n処理完了。特徴量抽出結果を '{output_file_path}' に保存しました。")
    print("\n--- 保存されたデータの出力例（先頭5件）---")
    print(df_to_save.head())
    
    return df_to_save

if __name__ == "__main__":
    # このスクリプトを直接実行した場合の例
    # 実際のパスに合わせて設定を調整してください
    from src.config import OUTPUT_FILE_CITING_PAPERS_WITH_PATHS, OUTPUT_DIR_PROCESSED
    ANNOTATION_LIST_CSV = '../data/ground_truth/annotation_target_list.csv'
    MASTER_LIST_CSV = OUTPUT_FILE_CITING_PAPERS_WITH_PATHS
    OUTPUT_FILE = 'features_for_evaluation.csv'

    try:
        df_targets = pd.read_csv(ANNOTATION_LIST_CSV)
        df_master = pd.read_csv(MASTER_LIST_CSV)
        process_xml_for_features(df_targets, df_master, OUTPUT_DIR_PROCESSED, OUTPUT_FILE)
    except FileNotFoundError as e:
        print(f"必要なファイルが見つかりません: {e}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
