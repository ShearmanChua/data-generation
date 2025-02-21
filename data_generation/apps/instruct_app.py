import os

import gradio as gr
import openai
import pandas as pd

from examples.instruct_examples import INSTRUCT_EXAMPLES
from pipelines.instruct_pipeline import InstructPipeline
from utils.utils import *


instruct_pipeline = InstructPipeline()

def generate_examples(dataset_description):
    examples = INSTRUCT_EXAMPLES[dataset_description]
    df = pd.DataFrame(examples)
    return df

def generate_sample_data(dataset_description, example_instructions, uploaded_files):
    num_examples = 5

    extracted_texts = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            assert uploaded_file.name.endswith(".pdf") or uploaded_file.name.endswith(".docx"), "Only PDF and DOCX files are supported."
            extracted_texts.append(extract_text_from_document(uploaded_file))
    
    extracted_nodes = [parse_markdown(text) for text in extracted_texts]





def generate_synthetic_data(instruction, num_samples, uploaded_files):
    extracted_text = ""
    if uploaded_files:
        for uploaded_file in uploaded_files:
            assert uploaded_file.name.endswith(".pdf") or uploaded_file.name.endswith(".docx"), "Only PDF and DOCX files are supported."
            extracted_text = extract_text_from_document(uploaded_file)

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
        gr.Markdown(value="## 1. Define your dataset")
        gr.Markdown(
            "To start, you may upload one or more documents, and provide a dataset description, as well as examples to generate synthetic fine-tuning data."
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                files_input = gr.File(
                    label="Upload document(s) (.pdf or .docx)",
                    file_types=[".pdf", ".docx"],
                    file_count="multiple",
                )
                with gr.Row():
                    clear_file_btn_part = gr.Button(
                        "Clear", variant="secondary"
                    )
                    load_file_btn = gr.Button("Load", variant="primary")

                dataset_description = gr.Textbox(
                    label="Dataset description (precise description of your desired dataset)",
                    placeholder="Give a precise description of your desired dataset.",
                )
                with gr.Row():
                    clear_prompt_btn_part = gr.Button("Clear", variant="secondary")
                    load_prompt_btn = gr.Button("Create", variant="primary")
            with gr.Column(scale=3):
                gr.Markdown("Example instructions and completion answers (click on the box to edit or add new rows):")
                example_instructions = get_dataframe(
                    columns=["instruction", "completion"],
                    kwargs={"wrap": True, "interactive": True},
                )

                default_examples = gr.Examples(
                    fn=generate_examples,
                    examples=[
                        ["Summarize news articles using 5W1H"],
                        ["Generate a customer support response"],
                        ["Translate a paragraph"],
                    ],
                    inputs=[dataset_description],
                    run_on_click=True,
                    outputs=[example_instructions],
                    cache_examples=False,
                    label="Examples",
                )

        gr.HTML(value="<hr>")
        gr.Markdown(value="## 2. Generated sample dataset")
        with gr.Row():
            system_prompt = gr.Textbox(label="System prompt",
                                       placeholder="System Prompt",
                                       interactive=False)
        with gr.Row(equal_height=False):
            sample_dataset = get_dataframe(columns=["instruction", "completion"])

        gr.HTML(value="<hr>")
        gr.Markdown(value="## 3. Generate synthetic fine-tuning data")
        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                num_samples_input = gr.Number(
                    label="Number of Samples", value=5, precision=0
                )
                with gr.Row():
                    generate_full_data_button = gr.Button("Generate Synthetic Data")

            with gr.Column(scale=3):
                output_text = get_dataframe(columns=["instruction", "completion"])

        generate_full_data_button.click(
            fn=generate_synthetic_data,
            inputs=[dataset_description, num_samples_input, files_input],
            outputs=output_text,
        )

        gr.Markdown(
            "**Tip:** Use these generated examples to fine-tune your model for better performance on specific tasks."
        )
