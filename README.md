# Attractiveness Prediction System

A learning system that predicts attractiveness scores (1-10 scale) using transfer learning on celebrity ELO ratings. Built for potential dating app integration to enable similarity-based matching.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Figures and Visualizations](#figures-and-visualizations)
- [Limitations](#limitations)
- [Blog Post](#blog-post)
- [Future Work](#future-work)

## Overview

This project implements a neural network that predicts facial attractiveness scores trained on a dataset of 1,000 celebrities rated through an ELO ranking system. The model uses transfer learning with ResNet18 and achieves strong performance across the full 1-10 attractiveness scale.

**Key Features:**
- Predicts attractiveness scores (1-10 scale)
- Command-line tool for single photo predictions
- Batch processing for multiple photos
- Dating app match recommendations based on similar scores
- Transfer learning for efficient training with limited data

**Potential Applications:**
- Dating app matching algorithms
- Profile photo optimization
- Social psychology research
- Computer vision applications

## Dataset

The dataset consists of 1,000 celebrity images rated through a head-to-head ELO ranking system where users voted on "who is more attractive?" in pairwise comparisons.

**Dataset Source:** [Add link to your dataset source]

**Original Data Challenges:**
- Average celebrity played only 3.6 games (98% had <10 games)
- Scores compressed to 1-3 range due to insufficient data
- Imbalanced gender distribution (726 female, 274 male)

**Data Preprocessing:**
- Percentile-based rescaling to utilize full 1-10 range
- Preserved relative rankings while expanding score distribution
- Final dataset: 1,000 celebrities with scores 1.1-10.0 (mean: 5.5, std: 2.6)

**Dataset Location:** `data/normalized_celebrity_ratings_rescaled.csv`

**Reproducibility:** 
All preprocessing steps are documented in `fix_low_games.py` and `analyze_elo.py`. The rescaled dataset used for training is included in this repository.

## Architecture

### Model Design

**Base Model:** ResNet18 (pretrained on ImageNet)
- Transfer learning leverages existing feature extraction capabilities
- Fine-tuned on attractiveness-specific features

**Custom Head:**
```python
nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1),
    nn.Sigmoid()
)
```

**Training Configuration:**
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam (lr=0.0005)
- Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)
- Early Stopping: Patience of 7 epochs
- Data Split: 80% train (800), 20% validation (200)

**Data Augmentation:**
- Random horizontal flip
- Random crop (256 to 224)
- Color jitter (brightness, contrast)
- Standard ImageNet normalization

### Architecture Decisions

**Why Single Model Instead of Gender-Specific?**
- Initial attempt: Separate male/female models
- Problem: Female model (726 examples) overfit, male model (274 examples) underfit
- Solution: Combined model with simpler architecture prevents overfitting
- Result: Better generalization across both genders

**Why Simplified Architecture?**
- Original complex head (512 to 256 to 64 to 1) overfit to training data
- Simpler architecture (512 to 128 to 1) with higher dropout prevents overfitting
- Early stopping prevents excessive memorization

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Approximately 2GB disk space for dependencies
- Approximately 500MB disk space for celebrity images (downloaded automatically)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/attractiveness-predictor.git
cd attractiveness-predictor
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Prepare the data:**
```bash
python setup.py
```

This verifies the dataset and checks score distribution.

4. **Train the model:**
```bash
python train_simple.py
```

**Training time:** Approximately 45-60 minutes on CPU (MacBook Air M1)  
**Output:** `attractiveness_model_simple.pth` (approximately 45MB)  
**Expected validation loss:** Approximately 0.08

**Note:** The trained model file is not included in this repository due to size constraints. Follow the training step above to generate it, or download from [Google Drive link if provided].

## Usage

### Command Line Prediction

Predict attractiveness score for a single photo:

```bash
python predict_simple.py --image path/to/photo.jpg
```

**Example output:**
```
======================================================================
  Attractiveness Score: 7.85 / 10
======================================================================
  Very attractive!
```

### Programmatic Usage

Use the model in your own Python scripts:

```python
from predict_simple import predict

# Predict single image
score = predict('photo.jpg')
print(f"Attractiveness score: {score:.2f}/10")

# Calculate dating app match range
min_match = max(1, score - 0.5)
max_match = min(10, score + 0.5)
print(f"Match range: {min_match:.1f} - {max_match:.1f}")
```

### Batch Processing

Process multiple photos:

```python
import glob
from predict_simple import predict

photos = glob.glob('photos/*.jpg')
scores = {}

for photo in photos:
    score = predict(photo)
    scores[photo] = score
    print(f"{photo}: {score:.2f}/10")

# Find best photo
best_photo = max(scores, key=scores.get)
print(f"\nBest photo: {best_photo} ({scores[best_photo]:.2f}/10)")
```

## Results

### Model Performance

| Metric | Value |
|--------|-------|
| Best Validation Loss | 0.0794 |
| Training Epochs | 8 (early stopped) |
| Prediction Range | 5.4 - 9.9 |
| Training Time | Approximately 45-60 minutes (CPU) |

### Sample Predictions

**High Attractiveness Celebrities (9.8-10.0 actual):**
- Predictions range: 5.4-9.9
- Mean prediction: 7.4
- Standard deviation: 1.9

**Observations:**
- Model successfully uses full 1-10 range
- Avoids compression seen in initial attempts
- Early stopping prevents overfitting

### Comparison to Original Approach

| Approach | Training Data | Prediction Range | Result |
|----------|--------------|------------------|---------|
| Original | Unscaled (mean=2.74) | 1.8-3.8 | Compressed predictions |
| Gender-Specific | Rescaled (mean=5.5) | 1.9-6.9 (M), 2.2-3.5 (F) | Female model collapsed |
| **Final (Simple)** | **Rescaled (mean=5.5)** | **5.4-9.9** | **Success** |

## Figures and Visualizations

This section explains how each figure in the repository was generated and links to the specific code that created it.

### Figure 1: Training History Curves

![Training History](results/training_simple.png)

**Description:** Shows training and validation loss over epochs, demonstrating model convergence and early stopping behavior.

**Generated by:** `train_simple.py` (lines 135-142)

**Code block:**
```python
# In train_simple.py, after training loop completes
plt.figure(figsize=(10, 5))
plt.plot(history['train'], label='Train')
plt.plot(history['val'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./training_simple.png')
```

**How to regenerate:**
```bash
python train_simple.py
# Automatically generates training_simple.png in project root
# Move to results/ folder: mv training_simple.png results/
```

**Interpretation:**
- Training loss decreases from 0.0746 to 0.0095
- Validation loss achieves best performance (0.0794) at epoch 1
- Early stopping triggered at epoch 8 when validation stops improving
- Indicates proper learning without severe overfitting

---

### Figure 2: Score Distribution Comparison

![Score Comparison](results/score_comparison.png)

**Description:** Compares original compressed scores versus rescaled scores, showing the data preprocessing transformation that was necessary for proper model training.

**Generated by:** `fix_low_games.py` (lines 150-183)

**Code block:**
```python
# In fix_low_games.py
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original distribution (compressed)
axes[0, 0].hist(df['label'], bins=30, color='red', alpha=0.7)
axes[0, 0].set_title(f'Original Scores\n(Mean: {df["label"].mean():.2f})')
axes[0, 0].set_xlabel('Score')
axes[0, 0].set_ylabel('Count')

# Rescaled percentile distribution (recommended approach)
axes[0, 2].hist(df['rescaled_percentile'], bins=30, color='green', 
                alpha=0.7, edgecolor='black', linewidth=1.5)
axes[0, 2].set_title(f'Percentile Rescaling (Recommended)')

# Score vs games played relationships
axes[1, 0].scatter(df['games_played'], df['label'], alpha=0.5)
axes[1, 1].scatter(df['games_played'], df['rescaled_percentile'], 
                   alpha=0.5, color='green')

plt.tight_layout()
plt.savefig('./score_rescaling_comparison.png', dpi=150)
```

**How to regenerate:**
```bash
python fix_low_games.py
# Generates score_rescaling_comparison.png
# Rename and move: mv score_rescaling_comparison.png results/score_comparison.png
```

**Interpretation:**
- Top-left: Original scores severely compressed (mean=2.74, narrow distribution)
- Top-right: Rescaled scores properly spread across 1-10 range (mean=5.5)
- Bottom panels: Show relationship between games played and scores
- Demonstrates why rescaling was critical - original data would have caused model to only predict 2-3

---

### Figure 3: ELO Analysis

![ELO Analysis](results/elo_analysis.png)

**Description:** Analyzes ELO rating quality issues including games played distribution, score versus games correlation, and confidence adjustments.

**Generated by:** `analyze_elo.py` (lines 126-162)

**Code block:**
```python
# In analyze_elo.py
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Games played distribution
axes[0, 0].hist(df['games_played'], bins=30, color='blue', alpha=0.7)
axes[0, 0].axvline(10, color='r', linestyle='--', label='10 games threshold')
axes[0, 0].set_xlabel('Games Played')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Games Played Distribution')
axes[0, 0].legend()

# Score vs games played scatter
axes[0, 1].scatter(df['games_played'], df['label'], alpha=0.5)
axes[0, 1].set_xlabel('Games Played')
axes[0, 1].set_ylabel('Normalized Score')
axes[0, 1].set_title('Score vs Games Played')
axes[0, 1].axvline(10, color='r', linestyle='--', alpha=0.5)

# Original vs confidence-adjusted scores
confidence = np.minimum(df['games_played'] / 20.0, 1.0)
df['adjusted_score'] = (confidence * df['label'] + 
                        (1 - confidence) * df['label'].mean())
axes[1, 0].scatter(df['label'], df['adjusted_score'], alpha=0.5)
axes[1, 0].plot([1, 10], [1, 10], 'r--', label='No change line')
axes[1, 0].set_title('Original vs Confidence-Adjusted Scores')

# Score distributions comparison
axes[1, 1].hist(df['label'], bins=30, alpha=0.7, label='Original')
axes[1, 1].hist(df['adjusted_score'], bins=30, alpha=0.7, label='Adjusted')
axes[1, 1].set_title('Score Distributions')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('./elo_analysis.png', dpi=150)
```

**How to regenerate:**
```bash
python analyze_elo.py
# Requires normalized_celebrity_ratings_2025-12-03_2.csv (original data)
# Generates elo_analysis.png
# Move: mv elo_analysis.png results/
```

**Interpretation:**
- Top-left: 98% of celebrities have fewer than 10 games (unreliable ratings)
- Top-right: No clear correlation between games played and scores (data quality issue)
- Bottom-left: Confidence adjustment regresses low-game scores toward mean
- Bottom-right: Effect of adjustment on score distribution
- Highlights the fundamental data quality problem and why percentile rescaling was chosen

---

## Figure Generation Summary

| Figure | Generated By | Command | Location |
|--------|-------------|---------|----------|
| Training History | `train_simple.py` | `python train_simple.py` | `results/training_simple.png` |
| Score Comparison | `fix_low_games.py` | `python fix_low_games.py` | `results/score_comparison.png` |
| ELO Analysis | `analyze_elo.py` | `python analyze_elo.py` | `results/elo_analysis.png` |

**To regenerate all figures:**
```bash
# 1. Data analysis and preprocessing
python fix_low_games.py          # Creates score_rescaling_comparison.png
python analyze_elo.py            # Creates elo_analysis.png

# 2. Model training
python train_simple.py           # Creates training_simple.png

# 3. Organize figures
mkdir -p results
mv score_rescaling_comparison.png results/score_comparison.png
mv elo_analysis.png results/
mv training_simple.png results/
```

---

## Limitations

### Technical Limitations
1. **Training Data Size:** Only 1,000 celebrities - more data would improve accuracy
2. **Low Game Counts:** 98% of celebrities had fewer than 10 ELO games, leading to unreliable initial ratings
3. **Gender Imbalance:** 73% female, 27% male in training data
4. **Image Quality:** Model performance depends on photo quality, lighting, and clarity

### Ethical Considerations
1. **Subjectivity:** Attractiveness is highly subjective and culturally dependent
2. **Bias:** Model reflects biases of specific raters in ELO system
3. **Reductionism:** Reduces complex human attractiveness to single number
4. **Harmful Use Cases:** Could reinforce harmful beauty standards or enable discrimination

### Scope Limitations
1. **Face-Only:** Does not account for body, style, personality, or chemistry
2. **Static Photos:** Real attractiveness includes movement, expression, context
3. **Cultural Specificity:** Trained on Western beauty standards
4. **No Demographics:** Does not account for age, ethnicity, or individual preferences

**Responsible Use:**
This system is intended for research and entertainment purposes only. A numerical score does not define human worth, and attractiveness is multidimensional, subjective, and culturally specific.

## Blog Post

**Read the full blog post explaining this project:**
[Link to your blog post - add this before submission]

The blog post provides a non-technical explanation of this project, covering:
- Introduction and problem statement
- Methodology and data challenges
- Results and findings
- Discussion of challenges and solutions
- Conclusions and lessons learned
- Ethical considerations

## Future Work

### Technical Improvements
- Collect more ELO data (20+ games per celebrity)
- Expand dataset to 5,000+ celebrities
- Balance gender distribution (50/50 split)
- Implement attention mechanisms to identify key facial features
- Multi-task learning (predict age, gender, ethnicity simultaneously)
- Ensemble methods combining multiple models

### Application Development
- Web interface for easy photo upload
- Mobile app (iOS/Android)
- Full-stack dating app integration
- Real-time matching algorithm
- Photo quality scoring and recommendations

### Research Directions
- Cross-cultural attractiveness variations
- Temporal changes (age progression)
- Explainable AI - which features matter most
- Bias detection and mitigation
- Comparison to human inter-rater reliability

## Technology Stack

- **Deep Learning:** PyTorch 2.0+
- **Computer Vision:** torchvision, PIL
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **ResNet Architecture:** He et al. (2015) - Deep Residual Learning for Image Recognition
- **Transfer Learning Approach:** ImageNet pretrained weights
- **ELO Rating System:** Arpad Elo (chess rating methodology)
- **Celebrity Dataset:** [Cite your data source]

## Author

**Juliana Vega Lara**
- Duke University - Study Abroad Copenhagen
- Course: Artificial Neural Networks
- Date: December 2025

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. *CVPR 2016*.
2. Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009). ImageNet: A Large-Scale Hierarchical Image Database. *CVPR 2009*.
3. Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present*. Arco Publishing.

## Issues and Contributing

Found a bug or have a suggestion? Please open an issue on GitHub.

This is an academic project, but contributions and feedback are welcome.
