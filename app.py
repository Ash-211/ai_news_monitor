import gradio as gr
import threading
import subprocess
import time
from src.api.main import app as custom_api

# 1. Start the Discord Bot in the background
def run_bot():
    print("Starting Discord Bot in background...")
    subprocess.run(["python", "src/bot/bot.py"])

threading.Thread(target=run_bot, daemon=True).start()

# 2. Create a dummy Gradio interface to satisfy Hugging Face's requirements
def greet(dummy):
    return "AI News Monitor Backend is Running!"

demo = gr.Interface(
    fn=greet, 
    inputs="text", 
    outputs="text",
    title="AI News Monitor",
    description="This is the backend API and Discord Bot for the AI News Monitor. The frontend is hosted separately on Vercel."
)

# 3. Mount our actual FastAPI application into the Gradio app
app = gr.mount_gradio_app(custom_api, demo, path="/gradio")

# Hugging Face Spaces will automatically find this 'app' object and run it with Uvicorn.
