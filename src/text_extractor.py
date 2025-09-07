import xml.etree.ElementTree as ET
import re

# XMLの名前空間の定義
NAMESPACES = {
    'ce': 'http://www.elsevier.com/xml/common/dtd',
    'ja': 'http://www.elsevier.com/xml/ja/dtd',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'core': 'http://www.elsevier.com/xml/svapi/article/dtd',
    'xocs': 'http://www.elsevier.com/xml/xocs/dtd'
}

def extract_abstract_robustly(root: ET.Element) -> str or None:
    """
    XMLのroot要素から、判明した全てのパターンを試してアブストラクトを抽出する。

    Args:
        root (ET.Element): XMLのルート要素。

    Returns:
        str or None: 抽出されたアブストラクトテキスト、または見つからなかった場合はNone。
    """
    try:
        # パターン1: 本文内の詳細なアブストラクト (`<ce:abstract class="author">`)
        paras = root.findall('.//ja:article/ja:head/ce:abstract[@class="author"]//ce:simple-para', NAMESPACES)
        if paras:
            text = ' '.join(p.text.strip() for p in paras if p.text)
            if text: return ' '.join(text.split())

        # パターン2: 一般的なアブストラクト (`<ce:abstract>`)
        # パターン1が成功しなかった場合のみ試行
        if not paras: # Check if paras from pattern 1 was empty
            paras = root.findall('.//ce:abstract/ce:abstract-sec//ce:simple-para', NAMESPACES)
            if paras:
                text = ' '.join(p.text.strip() for p in paras if p.text)
                if text: return ' '.join(text.split())

        # パターン3: メタデータ内のアブストラクト (`<dc:description>`)
        # パターン1, 2が成功しなかった場合のみ試行
        if not paras: # Check if paras from pattern 2 was empty
            description = root.find('.//core:coredata/dc:description', NAMESPACES)
            if description is not None and description.text:
                text = description.text.strip()
                if text: return ' '.join(text.split())
            
    except Exception:
        return None
    return None

def extract_full_text_robustly(root: ET.Element) -> str or None:
    """
    XMLのroot要素から、図表や不要セクション、数式などを除いたクリーンな全文テキストを抽出する。

    Args:
        root (ET.Element): XMLのルート要素。

    Returns:
        str or None: 抽出された全文テキスト、または見つからなかった場合はNone。
    """
    try:
        full_text_parts = []
        excluded_keywords = ['acknowledgement', 'references', 'bibliography', 'author contribution', 'competing interest', 'funding']

        # パターンA: 構造化された本文 (`<ja:body><ce:sections>`)
        body_sections = root.find('.//ja:body/ce:sections', NAMESPACES)
        if body_sections is not None:
            for section in body_sections.findall('.//ce:section', NAMESPACES):
                title_tag = section.find('./ce:section-title', NAMESPACES)
                sec_title = title_tag.text.strip().lower() if title_tag is not None and title_tag.text else ''
                
                if not any(keyword in sec_title for keyword in excluded_keywords):
                    for para in section.findall('./ce:para', NAMESPACES):
                        para_text_parts = [para.text.strip()] if para.text else []
                        for child in para:
                            if child.tag not in [f'{{{NAMESPACES["ce"]}}}formula', f'{{{NAMESPACES["ce"]}}}display']:
                                if child.tail: 
                                    para_text_parts.append(child.tail.strip())
                                # child.text も考慮に入れる
                                if child.text:
                                    para_text_parts.append(child.text.strip())
                        
                        clean_text = ' '.join(' '.join(part for part in para_text_parts if part).split())
                        if clean_text: full_text_parts.append(clean_text)
            if full_text_parts:
                return " ".join(full_text_parts)

        # パターンB: 非構造化テキスト (`<rawtext>`)
        raw_text_element = root.find('.//xocs:doc/xocs:rawtext', NAMESPACES)
        if raw_text_element is not None and raw_text_element.text:
            raw_text = raw_text_element.text
            return ' '.join(raw_text.split())

    except Exception:
        return None
    return None
