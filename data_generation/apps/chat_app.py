import os

import docx
import gradio as gr
import openai
import PyPDF2


def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with open(pdf_file.name, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        text = f"Error reading PDF: {str(e)}"
    return text


def extract_text_from_docx(docx_file):
    text = ""
    try:
        doc = docx.Document(docx_file.name)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        text = f"Error reading DOCX: {str(e)}"
    return text


def generate_synthetic_data(instruction, num_samples, uploaded_file):
    extracted_text = ""
    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            extracted_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            extracted_text = extract_text_from_docx(uploaded_file)

    if not extracted_text:
        return "Error: No text extracted from the document."

    try:
        responses = []
        for _ in range(num_samples):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for generating fine-tuning data.",
                    },
                    {
                        "role": "user",
                        "content": f"{instruction}\n\nDocument Content:\n{extracted_text}",
                    },
                ],
            )
            synthetic_example = response["choices"][0]["message"]["content"]
            responses.append(synthetic_example)
        return "\n\n".join(responses)
    except Exception as e:
        return f"Error: {str(e)}"


with gr.Blocks() as app:
    with gr.Column() as main_ui:
        gr.Markdown("# Synthetic Data Generator for Instruction Fine-Tuning")
        gr.Markdown(
            "Upload a document and provide an instruction to generate synthetic fine-tuning data."
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                file_input = gr.File(
                    label="Upload a document (.pdf or .docx)",
                    file_types=[".pdf", ".docx"],
                )
                instruction_input = gr.Textbox(
                    label="Instruction", placeholder="Enter your instruction here..."
                )
                num_samples_input = gr.Number(
                    label="Number of Samples", value=5, precision=0
                )
                generate_button = gr.Button("Generate Synthetic Data")
            with gr.Column(scale=3):
                examples = gr.Examples(
                    examples=[
                        "Summarize a news article",
                        "Generate a customer support response",
                        "Translate a paragraph",
                    ],
                    inputs=[instruction_input],
                    cache_examples=False,
                    label="Examples",
                )

        output_text = gr.Textbox(label="Generated Synthetic Data", lines=10)

        generate_button.click(
            fn=generate_synthetic_data,
            inputs=[instruction_input, num_samples_input, file_input],
            outputs=output_text,
        )

        gr.Markdown(
            "**Tip:** Use these generated examples to fine-tune your model for better performance on specific tasks."
        )
