from huggingface_hub import InferenceClient
import gradio as gr

client = InferenceClient(
    "mistralai/Mistral-7B-Instruct-v0.1"
)


def format_prompt(message):
    prompt = "<s>"
    prompt += """
        <s>[INST] <<SYS>>
        I'm using some computer vision models that extract some characteristics from a photo, such as objects, landscapes and texts.
        There is 2 predictors, Panoptic and OCR, your objective is to use these characteristics to describe the scene WITH FEW WORDS.
        You cant ask nothing, just describe the scene WITH FEW WORDS, be direct.
        <</SYS>>"""
    prompt += f"[INST] {message} [/INST]"
    return prompt


def generate(
    prompt, _, temperature=0.1, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0,
):
    temperature = float(temperature)
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = format_prompt(prompt)

    stream = client.text_generation(
        formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text
        yield output
    return output



gr.ChatInterface(
    fn=generate,
    chatbot=gr.Chatbot(show_label=False, show_share_button=False,
                       show_copy_button=True, likeable=True, layout="panel"),
    title="""Mistral 7B"""
).launch(show_api=False)
