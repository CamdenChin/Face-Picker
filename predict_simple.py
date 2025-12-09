"""
Predict with Advanced Attractiveness Model

This script loads a trained attractiveness model and uses it to predict
an attractiveness score (on a 1–10 scale) for a single input image.

Usage (from command line):

    python predict_advanced.py --image path/to/image.jpg --model path/to/model.pth
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse


class AttractivenessNet(nn.Module):
    """
    Neural network architecture for attractiveness prediction.

    - Uses ResNet18 as a backbone feature extractor.
    - Replaces the final fully-connected layer with a custom head:
        [global features] -> Linear -> ReLU -> Dropout -> Linear -> scalar score

    The model outputs a single real-valued score for each image.
    """

    def __init__(self):
        super(AttractivenessNet, self).__init__()

        # Load a ResNet18 architecture from torchvision.
        # `weights=None` means we are not loading pretrained ImageNet weights here,
        # because we expect to load our own fine-tuned weights from the checkpoint.
        self.backbone = models.resnet18(weights=None)

        # Get the dimensionality of the features that ResNet18 produces
        # right before its original fully connected (fc) layer.
        num_features = self.backbone.fc.in_features

        # Replace the original classification head (`fc`) with our custom regression head.
        # The new head:
        #   - takes `num_features` inputs (the global pooled features),
        #   - maps them to 128 hidden units with a Linear layer,
        #   - applies ReLU non-linearity,
        #   - applies Dropout(0.3) to help with regularization,
        #   - and finally outputs a single scalar (1 unit) through another Linear layer.
        #
        # IMPORTANT: There is NO sigmoid here. The output is a raw number.
        # Any mapping to a specific numeric range (like 1–10) happens later, outside the model.
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (Tensor): Input batch of images with shape [batch_size, 3, H, W].

        Returns:
            Tensor: Predicted scores with shape [batch_size, 1].
        """
        return self.backbone(x)


def predict(image_path, model_path='./attractiveness_model_simple.pth'):
    """
    Run a single-image prediction using a trained attractiveness model.

    Steps:
      1. Select device (GPU if available, else CPU).
      2. Initialize the model architecture.
      3. Load trained model weights from the checkpoint file.
      4. Apply the same image preprocessing used during training.
      5. Run a forward pass through the model to get a raw score.
      6. Convert the raw score (assumed 0–1 percentile) to a 1–10 scale.

    Args:
        image_path (str): Path to the input image file.
        model_path (str): Path to the saved model checkpoint (.pth file).

    Returns:
        float: Attractiveness score on a 1–10 scale.
    """

    # Choose device: if CUDA GPU is available, use it; otherwise, fall back to CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Initialize a fresh instance of the model architecture.
    #    This defines the layers but does NOT yet load the trained weights.
    model = AttractivenessNet()

    # 2) Load checkpoint from disk.
    #    `map_location=device` makes sure tensors are loaded onto the correct device
    #    (CPU or GPU) regardless of where they were originally trained/saved.
    checkpoint = torch.load(model_path, map_location=device)

    # 3) Load the model state (actual trained parameters) into the architecture.
    #    We assume the checkpoint dictionary contains a key 'model_state_dict'
    #    which stores the weights from training.
    model.load_state_dict(checkpoint['model_state_dict'])

    # 4) Move the model to the chosen device (GPU/CPU).
    model.to(device)

    # 5) Switch the model to evaluation mode.
    #    This disables behaviors like Dropout and BatchNorm updates so that
    #    inference is deterministic and matches validation behavior.
    model.eval()

    # 6) Define image transformations that match the training setup.
    #    - Resize to 352x352 (larger than typical 224x224) to keep more facial detail.
    #    - CenterCrop to 352x352 to ensure consistent size (here it's redundant,
    #      but keeps the pipeline robust if you later change sizes).
    #    - Convert to tensor so it becomes a PyTorch tensor in [C, H, W] format.
    #    - Normalize with ImageNet means and stds to match what ResNet expects.
    transform = transforms.Compose([
        transforms.Resize((352, 352)),
        transforms.CenterCrop((352, 352)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],  # mean for each RGB channel
            [0.229, 0.224, 0.225]   # std for each RGB channel
        )
    ])

    # 7) Load the input image from disk.
    #    `.convert('RGB')` forces the image into 3-channel RGB mode in case it's
    #    grayscale or has an alpha channel.
    img = Image.open(image_path).convert('RGB')

    # 8) Apply the same transformations used during training, then:
    #    - `unsqueeze(0)` adds a batch dimension (shape: [1, 3, H, W])
    #    - `.to(device)` moves the tensor to the same device as the model.
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 9) Run the model in no-grad context to avoid tracking gradients (faster,
    #    less memory, and we are not training here).
    with torch.no_grad():
        # `model(img_tensor)` returns a tensor of shape [1, 1]
        out = model(img_tensor).item()  # `.item()` extracts the scalar value

    # 10) Interpret the raw output `out` as a percentile in [0, 1].
    #     We then map this to a 1–10 attractiveness scale using:
    #       score_1_to_10 = out * 9 + 1
    #     Explanation:
    #       - If out = 0.0 → score = 1.0
    #       - If out = 1.0 → score = 10.0
    #       - Linear mapping between 0 and 1 to 1 and 10.
    score_1_to_10 = out * 9 + 1

    # Return the final human-readable attractiveness score.
    return score_1_to_10


def main():
    """
    Command-line entry point.

    - Parses arguments for image path and model path.
    - Calls `predict` to get the attractiveness score.
    - Prints a nicely formatted message plus a text label based on the score.
    """

    # Set up argument parser for CLI usage.
    parser = argparse.ArgumentParser()

    # Required argument: path to the image to be scored.
    parser.add_argument('--image', required=True)

    # Optional argument: path to the model checkpoint.
    # Defaults to './attractiveness_model_simple.pth'.
    parser.add_argument('--model', default='./attractiveness_model_simple.pth')

    # Parse arguments from the command line.
    args = parser.parse_args()

    try:
        # Run prediction with the given image and model paths.
        score = predict(args.image, args.model)

        # Print a visual separator for nicer CLI output.
        print("=" * 60)
        print(f"  Attractiveness Score: {score:.2f} / 10")
        print("=" * 60)

        # Print a qualitative label based on the numeric score.
        # These thresholds are arbitrary, just for UI/UX niceness.
        if score >= 8.0:
            print("  🌟 Very attractive!")
        elif score >= 6.5:
            print("  ✨ Attractive!")
        elif score >= 5.0:
            print("  😊 Above average")
        elif score >= 3.5:
            print("  👍 Average")
        else:
            print("  🙂 Below average")

    except Exception as e:
        # If anything goes wrong (e.g., invalid path, corrupt checkpoint, etc.),
        # print the error message instead of crashing silently.
        print(f"Error: {e}")


# Standard Python pattern to ensure that `main()` only runs when this script
# is executed directly (e.g., `python predict_advanced.py`) and NOT when it is
# imported as a module from another script.
if __name__ == "__main__":
    main()
