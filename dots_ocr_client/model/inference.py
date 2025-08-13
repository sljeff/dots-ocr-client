import json
import io
import base64
import math
from PIL import Image
import requests
from dots_ocr_client.utils.image_utils import PILimage_to_base64
from openai import OpenAI
import os


def inference_with_vllm(
        image,
        prompt, 
        ip="localhost",
        port=8000,
        temperature=0.1,
        top_p=0.9,
        max_completion_tokens=32768,
        model_name='model',
        ):
    
    addr = f"http://{ip}:{port}/v1"
    client = OpenAI(api_key="{}".format(os.environ.get("API_KEY", "0")), base_url=addr)
    messages = []
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url":  PILimage_to_base64(image)},
                },
                {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"}  # if no "<|img|><|imgpad|><|endofimg|>" here,vllm v1 will add "\n" here
            ],
        }
    )
    try:
        response = client.chat.completions.create(
            messages=messages, 
            model=model_name, 
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p)
        response = response.choices[0].message.content
        return response
    except requests.exceptions.RequestException as e:
        print(f"request error: {e}")
        return None


def inference_with_replicate(
        image,
        prompt,
        *,
        deployment: str | None = None,
        api_token: str | None = None,
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_completion_tokens: int = 2048,
    ):
    """
    Run inference via Replicate. If `deployment` is provided, use the deployment API; otherwise
    run the public model "sljeff/dots.ocr". Returns a JSON string for compatibility with
    post_process_output which expects JSON text for layout tasks.
    """
    # Lazy import so users who don't use replicate don't need the package installed
    try:
        import replicate
    except Exception as e:
        raise RuntimeError("Replicate backend selected but 'replicate' package is not installed."
                           " Install it or use backend='vllm'.") from e

    if api_token:
        os.environ["REPLICATE_API_TOKEN"] = api_token

    # Convert PIL.Image to an in-memory file object accepted by Replicate
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.name = "image.png"
    buf.seek(0)

    inputs = {
        "image": buf,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_completion_tokens),
    }

    if deployment:
        dep = replicate.deployments.get(deployment)
        pred = dep.predictions.create(input=inputs)
        pred.wait()
        output = pred.output
    else:
        # Use the public model by default for a clean UX
        output = replicate.run("sljeff/dots.ocr", input=inputs)

    # Return the JSON string directly - it's already a string from Replicate
    return output

