from pathlib import Path
from typing import List

import gradio as gr

from docling.backend.msword_backend import MsWordDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import FigureElement, InputFormat, Table
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc import ImageRef, ImageRefMode, PictureItem, TableItem
from pydantic import AnyUrl

IMAGE_RESOLUTION_SCALE = 2.0
pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
# pipeline_options.generate_page_images = True
# pipeline_options.generate_picture_images = True

doc_converter = (
    DocumentConverter(  # all of the below is optional, has internal defaults.
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
            InputFormat.ASCIIDOC,
            InputFormat.MD,
        ],  # whitelist formats, non-matching files are ignored.
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline, pipeline_options=pipeline_options
            ),
            InputFormat.DOCX: WordFormatOption(
                pipeline_cls=SimplePipeline, backend=MsWordDocumentBackend
            ),
        },
    )
)


def get_dataframe(
    columns: List[str] = ["prompt", "completion"],
    kwargs={"wrap": True, "interactive": False},
):
    return gr.Dataframe(headers=columns, **kwargs)


def extract_text_from_document(document):
    text = ""
    try:
        with open(document.name, "rb") as source:
            conversion_result = doc_converter.convert(source)
            markdown = conversion_result.document.export_to_markdown(
                image_mode=ImageRefMode.REFERENCED
            )
            text = markdown
    except Exception as e:
        text = f"Error reading file: {str(e)}"
    return text


import re


def parse_markdown(text):
    markdown_nodes = []
    lines = text.split("\n")
    current_section = ""
    # Keep track of headers at each level
    header_stack: List[str] = []
    code_block = False

    for line in lines:
        # Track if we're inside a code block to avoid parsing headers in code
        if line.lstrip().startswith("```"):
            code_block = not code_block
            current_section += line + "\n"
            continue

        # Only parse headers if we're not in a code block
        if not code_block:
            header_match = re.match(r"^(#+)\s(.*)", line)
            if header_match:
                # Save the previous section before starting a new one
                if current_section.strip():
                    markdown_nodes.append(
                        current_section.strip(),
                    )

                level = len(header_match.group(1))
                header_text = header_match.group(2)

                # Pop headers of equal or higher level
                while header_stack and len(header_stack) >= level:
                    header_stack.pop()

                # Add the new header
                header_stack.append(header_text)
                current_section = "#" * level + f" {header_text}\n"
                continue

        current_section += line + "\n"

    # Add the final section
    if current_section.strip():
        markdown_nodes.append(
            current_section.strip(),
        )

    return markdown_nodes
