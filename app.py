from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import torch
from torchvision import transforms
import os
from autoencoder import AutoEncoder  # Import your AutoEncoder

app = Flask(__name__)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)
model.load_state_dict(torch.load("saved_models/autoencoder.pth", map_location=device))
model.eval()

# Image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # Reverse normalization
    transforms.ToPILImage()
])

@app.route('/denoise', methods=['POST'])
def denoise_image():
    file = request.files['image']
    img = Image.open(file).convert('RGB')
    img = img.resize((256, 256))  # Resize to model's input size
    noisy_img = img.copy()

    # Add noise to the image
    img_tensor = transform(img).unsqueeze(0).to(device)
    noisy_tensor = img_tensor + 0.1 * torch.randn_like(img_tensor).to(device)
    noisy_tensor = torch.clamp(noisy_tensor, 0., 1.)

    # Pass through the model
    with torch.no_grad():
        output_tensor = model(noisy_tensor)

    # Convert tensors to images
    denoised_img = inverse_transform(output_tensor.squeeze(0).cpu())
    noisy_img = inverse_transform(noisy_tensor.squeeze(0).cpu())

    # Save results to a buffer
    output_buffer = io.BytesIO()
    denoised_img.save(output_buffer, format="JPEG")
    output_buffer.seek(0)

    return send_file(output_buffer, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
