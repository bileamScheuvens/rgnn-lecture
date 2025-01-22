import os
import re
from PIL import Image, ImageDraw, ImageFont

def create_collage(folder_path, output_path="collage.png", images_per_row=10):
    # Get all image files in the folder
    files = [f for f in os.listdir(folder_path) if re.match(r'sample_1_step_\d+\.png', f)]
    
    # Sort files in descending order based on the step number
    files.sort(key=lambda x: int(re.search(r'step_(\d+)', x).group(1)), reverse=True)
    
    # Load images and add captions
    images = []
    captions = []
    for file in files:
        step = re.search(r'step_(\d+)', file).group(1)
        img = Image.open(os.path.join(folder_path, file))
        images.append(img)
        captions.append(f"step {step}")
    
    # Determine image size and collage dimensions
    img_width, img_height = images[0].size
    collage_width = images_per_row * img_width
    collage_height = (len(images) // images_per_row + (len(images) % images_per_row > 0)) * (img_height + 30)
    
    # Create a blank collage
    collage = Image.new("RGB", (collage_width, collage_height), color="white")
    draw = ImageDraw.Draw(collage)
    
    # Use a default font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    # Add images and captions to the collage
    for i, (img, caption) in enumerate(zip(images, captions)):
        x = (i % images_per_row) * img_width
        y = (i // images_per_row) * (img_height + 30)
        collage.paste(img, (x, y))
        
        # Add caption below the image
        text_bbox = font.getbbox(caption)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x + (img_width - text_width) // 2
        text_y = y + img_height + 5
        draw.text((text_x, text_y), caption, fill="black", font=font)
    
    # Save the collage
    collage.save(output_path)
    print(f"Collage saved to {output_path}")

# Example usage
create_collage("./", output_path="collage.png")
