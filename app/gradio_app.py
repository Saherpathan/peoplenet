import gradio as gr
from main import process_frame


def video_stream():
    while True:
        grid = process_frame()
        yield grid

if __name__ == "__main__":
    gr.Interface(
        fn=video_stream,
        inputs=None,
        outputs=gr.Image(type="numpy", streaming=True),
        title="Live People Detection - 2x2 Feed",
        description="Displays 4 video streams with people detection only (conf > 0.72)"
    ).launch(share=True)
