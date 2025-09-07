import unittest
import xml.etree.ElementTree as ET
from src.text_extractor import extract_abstract_robustly, extract_full_text_robustly, NAMESPACES

class TestTextExtractor(unittest.TestCase):

    def setUp(self):
        # ダミーXMLコンテンツ
        self.dummy_xml_content_full = f"""
        <ja:article xmlns:ja="{NAMESPACES['ja']}" xmlns:ce="{NAMESPACES['ce']}" xmlns:dc="{NAMESPACES['dc']}" xmlns:core="{NAMESPACES['core']}" xmlns:xocs="{NAMESPACES['xocs']}">
            <ja:head>
                <ce:abstract class="author">
                    <ce:abstract-sec>
                        <ce:simple-para>This is the author's abstract. It describes the paper's content.</ce:simple-para>
                    </ce:abstract-sec>
                </ce:abstract>
                <ce:abstract>
                    <ce:abstract-sec>
                        <ce:simple-para>This is a general abstract.</ce:simple-para>
                    </ce:abstract-sec>
                </ce:abstract>
                <core:coredata>
                    <dc:description>This is a metadata abstract.</dc:description>
                </core:coredata>
            </ja:head>
            <ja:body>
                <ce:sections>
                    <ce:section>
                        <ce:section-title>Introduction</ce:section-title>
                        <ce:para>This is the introduction text.</ce:para>
                    </ce:section>
                    <ce:section>
                        <ce:section-title>Methods</ce:section-title>
                        <ce:para>Methodology details. <ce:formula>E=mc^2</ce:formula> More text.</ce:para>
                    </ce:section>
                    <ce:section>
                        <ce:section-title>Results</ce:section-title>
                        <ce:para>Results discussion. <ce:display>display_math</ce:display> Final text.</ce:para>
                    </ce:section>
                    <ce:section>
                        <ce:section-title>Acknowledgement</ce:section-title>
                        <ce:para>Thanks to everyone.</ce:para>
                    </ce:section>
                    <ce:section>
                        <ce:section-title>References</ce:section-title>
                        <ce:para>List of references.</ce:para>
                    </ce:section>
                </ce:sections>
            </ja:body>
            <xocs:doc>
                <xocs:rawtext>This is raw text content. It might be less structured.</xocs:rawtext>
            </xocs:doc>
        </ja:article>
        """
        self.root_full = ET.fromstring(self.dummy_xml_content_full)

        self.dummy_xml_no_abstract = f"""
        <ja:article xmlns:ja="{NAMESPACES['ja']}" xmlns:ce="{NAMESPACES['ce']}" xmlns:dc="{NAMESPACES['dc']}" xmlns:core="{NAMESPACES['core']}" xmlns:xocs="{NAMESPACES['xocs']}">
            <ja:head>
                <core:coredata>
                    <dc:title>No Abstract Paper</dc:title>
                </core:coredata>
            </ja:head>
            <ja:body>
                <ce:sections>
                    <ce:section>
                        <ce:section-title>Body</ce:section-title>
                        <ce:para>Some body text.</ce:para>
                    </ce:section>
                </ce:sections>
            </ja:body>
        </ja:article>
        """
        self.root_no_abstract = ET.fromstring(self.dummy_xml_no_abstract)

        self.dummy_xml_no_body = f"""
        <ja:article xmlns:ja="{NAMESPACES['ja']}" xmlns:ce="{NAMESPACES['ce']}" xmlns:dc="{NAMESPACES['dc']}" xmlns:core="{NAMESPACES['core']}" xmlns:xocs="{NAMESPACES['xocs']}">
            <ja:head>
                <ce:abstract class="author">
                    <ce:abstract-sec>
                        <ce:simple-para>Abstract only.</ce:simple-para>
                    </ce:abstract-sec>
                </ce:abstract>
            </ja:head>
        </ja:article>
        """
        self.root_no_body = ET.fromstring(self.dummy_xml_no_body)

    def test_extract_abstract_robustly_author_abstract(self):
        abstract = extract_abstract_robustly(self.root_full)
        self.assertEqual(abstract, "This is the author's abstract. It describes the paper's content.")

    def test_extract_abstract_robustly_general_abstract(self):
        # author abstractがない場合をシミュレート
        root_no_author_abstract = ET.fromstring(self.dummy_xml_content_full.replace('<ce:abstract class="author">', '<ce:abstract class="other">'))
        abstract = extract_abstract_robustly(root_no_author_abstract)
        self.assertEqual(abstract, "This is a general abstract.")

    def test_extract_abstract_robustly_metadata_abstract(self):
        # author abstractもgeneral abstractもない場合をシミュレート
        xml_content = self.dummy_xml_content_full.replace('<ce:abstract class="author">', '<ce:abstract class="other">')
        xml_content = xml_content.replace('<ce:abstract>', '<ce:abstract class="other_general">')
        root_no_other_abstracts = ET.fromstring(xml_content)
        abstract = extract_abstract_robustly(root_no_other_abstracts)
        self.assertEqual(abstract, "This is a metadata abstract.")

    def test_extract_abstract_robustly_no_abstract(self):
        abstract = extract_abstract_robustly(self.root_no_abstract)
        self.assertIsNone(abstract)

    def test_extract_full_text_robustly_structured_body(self):
        full_text = extract_full_text_robustly(self.root_full)
        expected_text_parts = [
            "This is the introduction text.",
            "Methodology details. More text.", # Formula excluded
            "Results discussion. Final text." # Display excluded
        ]
        expected_full_text = " ".join(expected_text_parts)
        self.assertEqual(full_text, expected_full_text)
        self.assertNotIn("Acknowledgement", full_text)
        self.assertNotIn("References", full_text)
        self.assertNotIn("E=mc^2", full_text)
        self.assertNotIn("display_math", full_text)

    def test_extract_full_text_robustly_raw_text(self):
        # structured bodyがない場合をシミュレート
        xml_content = self.dummy_xml_content_full.replace('<ja:body>', '<ja:body_other>')
        root_no_structured_body = ET.fromstring(xml_content)
        full_text = extract_full_text_robustly(root_no_structured_body)
        self.assertEqual(full_text, "This is raw text content. It might be less structured.")

    def test_extract_full_text_robustly_no_full_text(self):
        full_text = extract_full_text_robustly(self.root_no_body)
        self.assertIsNone(full_text)

    def test_extract_full_text_robustly_empty_body_sections(self):
        xml_content = f"""
        <ja:article xmlns:ja="{NAMESPACES['ja']}" xmlns:ce="{NAMESPACES['ce']}">
            <ja:body>
                <ce:sections/>
            </ja:body>
        </ja:article>
        """
        root = ET.fromstring(xml_content)
        full_text = extract_full_text_robustly(root)
        self.assertIsNone(full_text)

if __name__ == '__main__':
    unittest.main()
