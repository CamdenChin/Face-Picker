title: Face-Picker (Attractiveness Prediction)

purpose: |
  Predict attractiveness scores (1–10) using celebrity images and Elo-style
  attractiveness labels. Trains a deep model using percentile regression and
  pairwise ranking (RankNet). Allows prediction on any face image.

project_structure:
  - Face-Picker/
    - download_images.py
    - train_advanced.py
    - predict_advanced.py
    - setup.py
    - normalized_celebrity_ratings_rescaled.csv
    - celebrity_images/
    - attractiveness_advanced.pth  # created after training

workflow:
  step_1_setup:
    description: Install dependencies
    commands:
      - cd ~/Desktop/Face-Picker
      - python3 setup.py
  step_2_download_images:
    description: Download all celebrity images from URLs in the CSV
    commands:
      - python3 download_images.py
    result: "Approximately 1000 images saved to celebrity_images/"
  step_3_train_model:
    description: Train the advanced attractiveness model
    commands:
      - python3 train_advanced.py
    output:
      - attractiveness_advanced.pth
    notes: |
      Uses transfer learning with ResNet18, percentile regression,
      and pairwise ranking loss. Early stopping enabled.
  step_4_predict:
    description: Predict attractiveness score for a local image
    commands:
      - python3 predict_advanced.py --image ~/Desktop/photo.jpg
    example_output: |
      ============================================================
        Attractiveness Score: 7.42 / 10
      ============================================================

training_details:
  model: ResNet18
  techniques:
    - transfer_learning
    - percentile_regression
    - pairwise_ranking (RankNet)
    - no_sigmoid
    - higher_resolution_images
  training_time: "30–60 min on CPU (Mac), early stopping applies"
  batch_types:
    - single-image regression
    - pairwise comparison batches

important_files:
  - normalized_celebrity_ratings_rescaled.csv
  - celebrity_images/
  - attractiveness_advanced.pth

troubleshooting:
  images_not_found:
    symptom: "Images available: 0/1000"
    fix:
      - python3 download_images.py
      - confirm CSV has id and image_path columns
  model_not_found:
    symptom: "No such file attractiveness_advanced.pth"
    fix:
      - run training again

notes: |
  - Predictions are approximate and subjective.
  - Model trained on celebrities; non-celebrity photos may vary.
  - Use for fun, research, and experimentation.

disclaimer: |
  This system predicts attractiveness numerically, which is highly
  subjective and influenced by cultural and personal bias. Use responsibly.
