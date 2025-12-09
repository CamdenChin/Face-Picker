name: Face-Picker Setup Guide
version: 1.0
description: |
  Instructions to install dependencies, download dataset images,
  train the attractiveness model, and test prediction on an image.

requirements:
  python: "3.9+"
  operating_systems:
    - macOS
    - Linux
    - Windows
  internet: true

project_structure: |
  Face-Picker/
    setup.py
    train_simple.py
    predict_simple.py
    download_images.py
    normalized_celebrity_ratings_rescaled.csv
    celebrity_images/
      <1000 downloaded .jpg files>
    attractiveness_model_simple.pth
    training_simple.png

steps:

  - step: install_dependencies
    instructions: |
      Open Terminal and run:
        cd ~/Desktop/Face-Picker
        python3 setup.py
      This installs torch, torchvision, pandas, pillow, matplotlib, tqdm.
  
  - step: download_images
    instructions: |
      Install requests (if needed):
        pip install requests
      Then run:
        python3 download_images.py
      This reads image_path from the CSV and saves each image as:
        celebrity_images/<id>.jpg
      Expect ~1000 files downloaded.

  - step: train_model
    instructions: |
      Run:
        python3 train_simple.py
      You should see:
        Images available: 1000/1000
        Epoch...
        → Saved (val_loss: ...)
      After training, check that:
        attractiveness_model_simple.pth
      and:
        training_simple.png
      exist in the project folder.

  - step: test_prediction
    instructions: |
      Once the .pth model exists, test your own image:
        python3 predict_simple.py --image ~/Desktop/camden_prom.png

    expected_output: |
      ============================================================
        Attractiveness Score: 7.34 / 10
      ============================================================
        ✨ Attractive!

troubleshooting:

  missing_model_file:
    error_message: |
      Error: No such file or directory: './attractiveness_model_simple.pth'
    fix: |
      You must train first:
        python3 train_simple.py
      Or place the .pth file in the same directory as predict_simple.py.

  images_not_found:
    error_message: |
      Images available: 0/1000
    fix: |
      Run:
        python3 download_images.py
      Then verify:
        ls celebrity_images
      You should see .jpg files named by the id column.

notes: |
  • The label column in the CSV already contains attractiveness ratings
    (even if originally from Google Sheets).
  • The model learns directly from those values.
  • You only need to train once.
  • After training, you only need predict_simple.py + the .pth file
    to run predictions anywhere.

summary:
  - python3 setup.py
  - python3 download_images.py
  - python3 train_simple.py
  - python3 predict_simple.py --image <your_image_path>

done: |
  You can now rate faces with your custom model.
