import os
import sys
import gradio as gr
import math
import matplotlib.pyplot as plt

import requests
import fileinput
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import gradio as gr
import json
import math
import requests
from dotenv import load_dotenv, dotenv_values
# loading variables from .env file
load_dotenv()

vidOut = "results/results"
uvqOut = "results/modified_prompts_eval"
evalOut = "evaluation_results"
num_of_vid = 3
vid_length = 2
uvq_threshold = 3.8
fps = 24


# Generate the scores in csv files
def genScore():
    for i in range(1, num_of_vid+1):
        fileindex = f"{i:04d}"
        os.system(
            f'python3 ./uvq/uvq_main.py --input_files="{fileindex},2, {vidOut}/{fileindex}.mp4" --output_dir {uvqOut} --model_dir ./uvq/models'
        )


def getScore(filename):
    # MOS_score defines the output of the uvq score
    lines = str(filename).split('\n')
    last_line = lines[-1]
    MOS_score = last_line.split(',')[-1]
    MOS_score = MOS_score[:-2]

    return MOS_score

# MOS_score defines the Mean Opinion Score of prediction, if the video's MOS exceeds the threshold then we directly use this video


def chooseBestVideo():
    MOS_score_high = 0
    preferred_output = ""
    chosen_idx = 0

    for i in range(1, num_of_vid+1):
        '''We loop thru this current processed video'''
        filedir = f"{i:04d}"
        filename = f"{i:04d}_uvq.csv"
        with open(os.path.join(uvqOut, filedir, filename), 'r') as file:
            MOS = file.read().strip()

        MOS_score = getScore(MOS)
        print("Video Index:", f"{i:04d}", "Score:", MOS_score)

        # if the MOS_score is higher than the previous video, we choose this video as our preferred video output
        if float(MOS_score) > float(MOS_score_high) or float(MOS_score) > uvq_threshold:
            MOS_score_high = MOS_score
            preferred_output = filename
            chosen_idx = i

        if float(MOS_score) > uvq_threshold:
            break
    return chosen_idx
    # print(MOS_score_high)
    # print(preferred_output)


def extract_scores_from_json(json_path):
    with open(json_path) as file:
        data = json.load(file)

    for key, value in data.items():
        if isinstance(value, list) and len(value) > 1 and isinstance(value[0], float):
            motion_score = value[0]

    return motion_score


def VBench_eval(vid_filename):
    # vid_filename: video filename without .mp4
    os.system(
        f'python VBench/evaluate.py --dimension "motion_smoothness"  --videos_path {os.path.join(vidOut, vid_filename)}.mp4 --custom_input --output_filename {vid_filename}'
    )
    eval_file_path = os.path.join(
        evalOut, f"{vid_filename}_eval_results.json")
    motion_score = extract_scores_from_json(eval_file_path)

    return motion_score


def interpolation(chosen_idx, fps):
    vid_filename = f"{chosen_idx:04d}.mp4"
    os.chdir("ECCV2022-RIFE")
    os.system(
        f'python3 inference_video.py --exp=2 --video={os.path.join(vidOut, vid_filename)} --fps {fps}'
    )
    os.chdir("../")
    out_name = f"{chosen_idx:04d}_4X_{fps}fps.mp4"
    return out_name

# call the GPT API here


def call_gpt_api(prompt, isSentence=False):
    api_key = os.getenv("MY_GPT_KEY")

    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        },
        json={
            'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': prompt}],
            'model': 'gpt-3.5-turbo',
            # 'prompt': prompt,
            'temperature': 0.4,
            'max_tokens': 200
        })
    response_json = response.json()
    choices = response_json['choices']
    contents = [choice['message']['content'] for choice in choices]
    contents = [
        sentence for sublist in contents for sentence in sublist.split('\n')]
    # Remove the leading number and dot from each sentence
    sentences = [content.lstrip('1234567890.- ') for content in contents]
    if len(sentences) > 2 and isSentence:
        sentences = sentences[1:]
    return sentences


# Initialize Firebase Admin SDK
cred = credentials.Certificate(
    "final-year-project-443dd-df6f48af0796.json")
firebase_admin.initialize_app(cred)
# Initialize Firestore client
db = firestore.client()


def retrieve_user_feedback():
    # Retrieve user feedback from Firestore
    feedback_collection = db.collection("user_feedbacks")
    feedback_docs = feedback_collection.get()

    feedback_text = []
    experience = []
    for doc in feedback_docs:
        data = doc.to_dict()
        feedback_text.append(data.get('feedback_text', None))
        experience.append(data.get('experience', None))

    return feedback_text, experience


feedback_text, experience = retrieve_user_feedback()
# print("Feedback Text:", feedback_text)
# print("Experience:", experience)


def store_user_feedback(feedback_text, experience):
    # Get a reference to the Firestore collection
    feedback_collection = db.collection("user_feedbacks")

    # Create a new document with feedback_text and experience fields
    feedback_collection.add({
        'feedback_text': feedback_text,
        'experience': experience
    })
    return


t2v_examples = [
    ['A tiger walks in the forest, photorealistic, 4k, high definition'],
    ['an elephant is walking under the sea, 4K, high definition'],
    ['an astronaut riding a horse in outer space'],
    ['a monkey is playing a piano'],
    ['A fire is burning on a candle'],
    ['a horse is drinking in the river'],
    ['Robot dancing in times square'],
]


def generate_output(input_text, output_video_1, fps, examples):
    def generate_output_fn(input_text, output_video_1, fps, examples):
        if input_text == "":
            return input_text, output_video_1, examples
        output = call_gpt_api(
            prompt=f"Generate 2 similar prompts and add some reasonable words to the given prompt and not change the meaning, each within 30 words: {input_text}", isSentence=True)
        output.append(input_text)
        with open("prompts/test_prompts.txt", 'w') as file:
            for i, sentence in enumerate(output):
                if i < len(output) - 1:
                    file.write(sentence + '\n')
                else:
                    file.write(sentence)
        os.system(
            f'sh {os.path.join("scripts", "run_text2video.sh")}')
        # Connect the video output and return the video corresponding link
        genScore()
        chosen_idx = chooseBestVideo()
        chosen_vid_path = interpolation(chosen_idx, fps)
        chosen_vid_path = f"{vidOut}/{chosen_vid_path}"
        output_video_1 = gr.Video(
            value=chosen_vid_path, show_download_button=True)

        examples_list = call_gpt_api(
            prompt=f"Generate 5 similar prompts that makes a storyline coming after the given input, each within 10 words: {input_text}")
        examples = []
        for prompt in examples_list:
            examples.append([prompt])
        input_text = ""

        return input_text, output_video_1, examples

    return generate_output_fn(input_text, output_video_1, fps, examples)


def t2v_demo(result_dir='./tmp/'):
    with gr.Blocks() as videocrafter_iface:
        gr.Markdown("<div align='center'> <h2> VideoCraftXtend: AI-Enhanced Text-to-Video Generation with Extended Length and Enhanced Motion Smoothness </span> </h2> </div>")

        # Initialize values for video length and fps
        video_len_value = 5.0

        def update_fps(video_len, fps):
            fps_value = 80 / video_len
            return f"<div justify-content: 'center'; text-align='center'> <h6> FPS (frames per second) : {int(fps_value)} </span> </h6> </div>"

        def load_example(example_id):
            return example_id[0]

        def update_feedback(value, text):
            labels = ['Positive', 'Neutral', 'Negative']
            colors = ['#66c2a5', '#fc8d62', '#8da0cb']
            if value != '':
                store_user_feedback(value, text)
                user_satisfaction.append(value)
                value = ''
            if text != '':
                user_feedback.append(text)
                text = ''
            user_feedback, user_satisfaction = retrieve_user_feedback()
            sizes = [user_satisfaction.count('Positive'), user_satisfaction.count(
                'Neutral'), user_satisfaction.count('Negative')]
            plt.pie(sizes, labels=labels, autopct='%1.1f%%',
                    startangle=140, colors=colors)
            plt.axis('equal')
            return plt

        with gr.Tab(label="Text2Video"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        input_text = gr.Text(
                            placeholder=t2v_examples[2], label='Please input your prompt here.')
                        with gr.Row():
                            examples = gr.Dataset(samples=t2v_examples, components=[
                                                  input_text], label='Sample prompts that can be used to form a storyline.')
                        with gr.Column():
                            gr.Markdown(
                                "<div align='center'> <h4> Modify video length and the corresponding fps will be shown on the right. </span> </h4> </div>")
                            with gr.Row():
                                video_len = gr.Slider(minimum=4.0, maximum=10.0, step=1, label='Video Length',
                                                      value=video_len_value, elem_id="video_len", interactive=True)
                                fps = gr.Markdown(
                                    elem_id="fps", value=f"<div> <h6> FPS (frames per second) : 16</span> </h6> </div>")
                        send_btn = gr.Button("Send")
                    with gr.Column():
                        with gr.Tab(label='Result'):
                            with gr.Row():
                                output_video_1 = gr.Video(
                                    value="sample/0009.mp4", show_download_button=True)

            video_len.change(update_fps, inputs=[video_len, fps], outputs=fps)
            # fps.change(update_video_len_slider, inputs = fps, outputs = video_len)

            examples.click(load_example, inputs=[
                           examples], outputs=[input_text])
            send_btn.click(
                fn=generate_output,
                inputs=[input_text, output_video_1, fps, examples],
                outputs=[input_text, output_video_1, examples],
            )

        with gr.Tab(label="Feedback"):
            with gr.Column():
                with gr.Column():
                    with gr.Row():
                        feedback_value = gr.Radio(
                            ['Positive', 'Neutral', 'Negative'], label="How is your experience?")
                        feedback_text = gr.Textbox(
                            placeholder="Enter feedback here", label="Feedback Text")
                    with gr.Row():
                        cancel_btn = gr.Button("Clear")
                        submit_btn = gr.Button("Submit")
                with gr.Row():
                    pie_chart = gr.Plot(value=update_feedback(
                        '', ''), label="Feedback Pie Chart")
                    with gr.Column():
                        gr.Markdown(
                            "<div align='center'> <h4> Feedbacks from users: </span> </h4> </div>")
                        feedback_text_display = [gr.Markdown(
                            feedback, label="User Feedback") for feedback in retrieve_user_feedback()[0]]
                submit_btn.click(fn=update_feedback, inputs=[
                                 feedback_value, feedback_text], outputs=[pie_chart])

    return videocrafter_iface


if __name__ == "__main__":
    result_dir = os.path.join('./', 'results')
    t2v_iface = t2v_demo(result_dir)
    t2v_iface.queue(max_size=10)
    t2v_iface.launch(debug=True)
