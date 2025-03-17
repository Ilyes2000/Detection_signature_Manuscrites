import gradio as gr
import numpy as np
from PIL import Image
import time
import cv2


import altair as alt
import pandas as pd

from signature_detector import SignatureDetector  



MODEL_PATH = "Models/llest.onnx"

detector = SignatureDetector(model_path=MODEL_PATH)


inference_times = []


def get_distribution_plot():
    """
    Renvoie un histogramme Altair de la distribution des temps d'inférence.
    """
    if len(inference_times) == 0:
        return alt.Chart(pd.DataFrame({"time": []}))  
    df = pd.DataFrame({"time": inference_times})
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            alt.X("time", bin=alt.Bin(maxbins=20), title="Time (ms)"),
            alt.Y("count()", title="Frequency"),
        )
        .properties(width=400, height=250, title="Distribution of Inference Times")
    )
    return chart

def get_time_history_plot():
    """
    Renvoie un graphique Altair montrant l’évolution du temps d’inférence.
    """
    if len(inference_times) == 0:
        return alt.Chart(pd.DataFrame({"index": [], "time": []}))
    
    df = pd.DataFrame({
        "index": range(1, len(inference_times) + 1),
        "time": inference_times
    })
    
    line = alt.Chart(df).mark_line().encode(
        x=alt.X("index", title="Inference Number"),
        y=alt.Y("time", title="Time (ms)"),
        tooltip=["index", "time"]
    )

    mean_time = np.mean(inference_times)
    rule = alt.Chart(pd.DataFrame({"mean": [mean_time]})).mark_rule(color="red").encode(
        y="mean"
    )

    chart = (
        (line + rule)
        .properties(width=400, height=250, title="Inference Time per Execution")
    )
    return chart


def detect_signatures(image, conf_threshold, iou_threshold):
    """
    Détection ONNX via SignatureDetector.
    
    Retourne :
      - Image annotée (PIL)
      - Texte d'info
      - Nombre total d'inférences (str)
      - Histogramme Altair
      - Historique Altair
      - Temps moyen (str)
      - Dernier temps d'inférence (str)
    """

    if image is None:
        return (None, "Aucune image fournie.", "0", None, None, "0", "0")

    
    start_time = time.time()

    output_image_np, metrics = detector.detect(
        image, conf_thres=conf_threshold, iou_thres=iou_threshold
    )
    
    end_time = time.time()
    inf_time = (end_time - start_time) * 1000.0  

    
    inference_times.append(inf_time)

   
    annotated_img = Image.fromarray(output_image_np)

  
    infos = (
        f"Temps d'inférence : {inf_time:.2f} ms\n"
        f"Voir l'image annotée ci-contre."
    )

   
    total_inferences = len(inference_times)
    avg_time = np.mean(inference_times)
    last_time = inf_time
    
    total_inferences_str = str(total_inferences)
    avg_time_str = f"{avg_time:.2f}"
    last_time_str = f"{last_time:.2f}"

    dist_plot = get_distribution_plot()
    history_plot = get_time_history_plot()

    return (
        annotated_img,
        infos,
        total_inferences_str,
        dist_plot,
        history_plot,
        avg_time_str,
        last_time_str
    )

def clear_all():
    """
    Fonction pour vider les champs du UI Gradio.
    """
    return None, None, "", "0", None, None, "0", "0"


with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>Détection de Signatures ONNX</h1>")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Image d'entrée", type="pil")
            conf_slider = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="Confidence Threshold")
            iou_slider = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="IoU Threshold")
            btn_detect = gr.Button("Detect", variant="primary")
            btn_clear = gr.Button("Clear")
        with gr.Column():
            result_image = gr.Image(label="Image annotée")
            result_text  = gr.Textbox(label="Résultats", lines=6)

    gr.Markdown("---")
    
    with gr.Row():
        total_inf_box = gr.Textbox(label="Total Inferences", value="0")
    
    with gr.Row():
        with gr.Column():
            time_dist_plot = gr.Plot(label="Time Distribution")
        with gr.Column():
            time_history_plot = gr.Plot(label="Time History")
    
    with gr.Row():
        avg_time_box = gr.Textbox(label="Average Inference Time (ms)", value="0")
        last_time_box = gr.Textbox(label="Last Inference Time (ms)", value="0")

    
    btn_detect.click(
        fn=detect_signatures,
        inputs=[image_input, conf_slider, iou_slider],
        outputs=[
            result_image,       
            result_text,        
            total_inf_box,      
            time_dist_plot,     
            time_history_plot,  
            avg_time_box,       
            last_time_box      
        ]
    )

    btn_clear.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            image_input,
            result_image,
            result_text,
            total_inf_box,
            time_dist_plot,
            time_history_plot,
            avg_time_box,
            last_time_box
        ]
    )

    demo.launch()
