# Import Key Libraries
from glob import glob
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import streamlit as st
import matplotlib.pyplot as plt

# --------------------------- Configuration ---------------------------
IMG_SIZE = 224
CONF_THRESHOLD = 4.5 # Confidence threshold (%)

# Set device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

# Load class names from training folder structure
base_dir = "./data"
train_dir = os.path.join(base_dir, "train")
train_dataset = datasets.ImageFolder(train_dir)
class_names = [name.split('-')[-1].lower() for name in train_dataset.classes]
num_classes = len(class_names)

# Define inference transforms
inference_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load model architecture and weights
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load("resnet50_best.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# --------------------------- Streamlit UI ---------------------------
st.title("🐶 Dog Breed Identifier")
st.markdown("Upload one or more images of dogs to predict their breed.")

uploaded_files = st.file_uploader("Choose JPG/JPEG images", type=["jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(min(3, len(uploaded_files)))  # Dynamic columns for layout
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file).convert("RGB")
        input_tensor = inference_transforms(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)

        pred_label = (class_names[pred.item()])
        confidence = conf.item() * 100
        color = "✅" if confidence >= CONF_THRESHOLD else "✳️☑️"

        # Unnormalize for display
        img_disp = input_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        img_disp = img_disp * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img_disp = img_disp.clip(0, 1)

        with cols[idx % len(cols)]:
            # st.image(img_disp, caption=f"{color} **{pred_label}**\n{confidence:.2f}%", use_container_width=True)
            with cols[idx % len(cols)]:
                st.image(img_disp, width=200)  # Shrink image size

                caption_html = f"""
                <div style="text-align: center;">
                    <span style="font-size: 16px; font-weight: bold;">{color} {pred_label.upper()}</span><br>
                    <span style="font-size: 14px; color: gray;">Confidence: {confidence:.2f}%</span>
                </div>
                """
                st.markdown(caption_html, unsafe_allow_html=True)

