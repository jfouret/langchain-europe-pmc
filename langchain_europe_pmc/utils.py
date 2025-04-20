"""EuropePMC XML utilities."""

import warnings
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from html import unescape

import requests
from bs4 import XMLParsedAsHTMLWarning
from markdownify import markdownify as md
from pydantic import BaseModel, Field


class EuropePMCXMLParser:
    def __init__(self, pmcid: str):
        self.pmcid = pmcid
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

    def _extract_keywords(self, root: ET.Element) -> str:
        """
        Extract keywords from the XML.

        Args:
            root: XML root element

        Returns:
            str: Markdown formatted keywords section
        """
        keywords = []
        kwd_group = root.find(".//kwd-group")

        if kwd_group is not None:
            for kwd in kwd_group.findall(".//kwd"):
                if kwd.text:
                    keywords.append(kwd.text.strip())

        if keywords:
            return "## Keywords\n" + ", ".join(keywords) + "\n\n"
        return ""

    def _element_to_html_string(self, element: ET.Element) -> str:
        """
        Convert an XML element to an HTML string.

        Args:
            element: XML element

        Returns:
            str: HTML string representation
        """
        # Convert ElementTree element to string
        xml_str = ET.tostring(element, encoding="unicode")

        # Parse with minidom for better formatting
        dom = minidom.parseString(xml_str)

        # Return the HTML string
        return dom.toxml()

    def _process_table_wrap(self, table_wrap: ET.Element) -> str:
        """
        Process a table-wrap element and convert it to markdown.

        Args:
            table_wrap: table-wrap element

        Returns:
            str: Markdown formatted table with caption
        """
        content = []

        # Process label
        label = table_wrap.find("./label")
        if label is not None:
            label_text = "".join(label.itertext()).strip()
            if label_text:
                content.append(f"**{label_text}**")

        # Process caption
        caption = table_wrap.find(".//caption/p")
        if caption is not None:
            caption_text = "".join(caption.itertext()).strip()
            if caption_text:
                content.append(f"*{caption_text}*")

        # Process table using markdownify
        table = table_wrap.find(".//table")
        if table is not None:
            # Convert table element to HTML string
            html_str = self._element_to_html_string(table)

            # Convert HTML to markdown using markdownify
            table_md = md(html_str)

            if table_md:
                content.append(table_md)

        return "\n\n".join(content)

    def _process_section(self, section: ET.Element, level: int = 2):
        """
        Process a section element and its children recursively.

        Args:
            section: Section element
            level: Heading level (default: 2 for ##)

        Returns:
            str: Markdown formatted section content
        """
        content = []

        # Process section title
        title = section.find("./title")
        if title is not None:
            title_text = "".join(title.itertext()).strip()
            if title_text:
                heading = "#" * level
                content.append(f"{heading} {title_text}\n")

        # Process paragraphs directly in this section
        for p in section.findall("./p"):
            p_text = "".join(p.itertext()).strip()
            if p_text:
                content.append(p_text)

        # Process tables
        for table_wrap in section.findall(".//table-wrap"):
            table_md = self._process_table_wrap(table_wrap)
            if table_md:
                content.append(table_md)

        # Process figures
        for fig in section.findall(".//fig"):
            # Process figure label
            label = fig.find("./label")
            if label is not None:
                label_text = "".join(label.itertext()).strip()
                if label_text:
                    content.append(f"**{label_text}**")

            # Process figure caption
            caption = fig.find(".//caption/p")
            if caption is not None:
                caption_text = "".join(caption.itertext()).strip()
                if caption_text:
                    content.append(f"*{caption_text}*")

        # Process subsections recursively
        for subsection in section.findall("./sec"):
            content.append(self._process_section(subsection, level + 1))

        return "\n\n".join(content)

    def extract_main_text_as_markdown(self) -> str:
        """
        Extract the main text content from an Europe PMC XML file and format as markdown.

        Args:
            pmcid (str): PubMed Central ID

        Returns:
            str: The extracted main text formatted as markdown
        """
        pmcid = self.pmcid
        # Construct the URL
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

        # Fetch the XML content
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            xml_str = response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching XML: {e}")
            return ""

        # Parse the XML content
        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            return ""

        # Extract keywords
        markdown_content = self._extract_keywords(root)

        # Find the body element
        body = root.find(".//body")
        if body is None:
            print("No body element found in the XML")
            return markdown_content

        # Process each top-level section
        for section in body.findall("./sec"):
            markdown_content += self._process_section(section) + "\n\n"

        # Handle any paragraphs directly in the body (not in sections)
        for p in body.findall("./p"):
            p_text = "".join(p.itertext()).strip()
            if p_text:
                markdown_content += p_text + "\n\n"

        # Clean up any HTML entities
        markdown_content = unescape(markdown_content)

        return markdown_content.strip()
