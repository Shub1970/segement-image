from PIL import Image
import numpy as np
from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation

# Load the image
image = Image.open("./new_bild_jpg.jpg")

# Load the model and process the image
processor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-large-ade")
inputs = processor(images=image, return_tensors="pt")

model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-ade")
outputs = model(**inputs)

# Post-process the semantic segmentation
predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

# Convert the predicted semantic map to NumPy array
predicted_semantic_map_np = predicted_semantic_map.cpu().numpy()

# Iterate through each class label and create a mask
for class_label in range(predicted_semantic_map_np.max() + 1):
    # Create a mask for the current class label
    mask = (predicted_semantic_map_np == class_label).astype(np.uint8) * 255

    # Check if the mask is not all zeros
    if np.any(mask):
        # Invert the mask: white -> black, black -> transparent
        inverted_mask = 255 - mask

        # Create a PIL Image from the inverted mask
        inverted_mask_image = Image.fromarray(inverted_mask)

        # Convert pixels other than black to transparent
        inverted_mask_image = inverted_mask_image.convert("RGBA")
        datas = inverted_mask_image.getdata()
        newData = []
        for item in datas:
            if item[0] != 0 or item[1] != 0 or item[2] != 0:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        inverted_mask_image.putdata(newData)

        # Save the inverted mask image
        inverted_mask_image.save(f"mask_{class_label}.png")
