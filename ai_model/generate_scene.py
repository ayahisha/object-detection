from diffusers import StableDiffusionPipeline
import torch

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to("cuda")

def generate_historical_scene(prompt, output_path="historical_scene.png"):
    image = pipe(prompt).images[0]
    image.save(output_path)
    print(f"Generated image saved at {output_path}")

# Example prompt
if __name__ == "__main__":
    prompt = "A historical 19th-century town with cobblestone streets and old buildings"
    generate_historical_scene(prompt)
