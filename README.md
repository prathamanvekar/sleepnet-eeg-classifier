# SleepNet EEG Classifier

Deep learning model for automated sleep stage classification from EEG signals using CNN, Temporal Convolutional Networks (TCN), and Attention mechanisms.

## Overview

SleepNet is a neural network architecture designed to classify sleep stages from EEG (electroencephalogram) data. The model automatically detects and classifies five sleep stages:

- **W** - Wake
- **N1** - Non-REM Stage 1
- **N2** - Non-REM Stage 2
- **N3** - Non-REM Stage 3 (Deep Sleep)
- **REM** - Rapid Eye Movement

## Features

- **Advanced Architecture**: Combines CNN for initial feature extraction, TCN for temporal pattern recognition, and Attention mechanism for important signal regions
- **EEG Signal Processing**: Automated preprocessing with bandpass filtering and normalization
- **Class Balancing**: Weighted random sampling for imbalanced dataset handling
- **Web Interface**: Flask-based interface for easy predictions on new EEG data
- **Evaluation Metrics**: F1-score, classification reports, and confusion matrices

## Project Structure

```
.
├── model.py              # SleepNet architecture definition
├── dataset.py            # EEG dataset loader
├── train.py              # Training pipeline
├── evaluate.py           # Model evaluation
├── preprocess.py         # Data preprocessing utilities
├── app.py                # Flask web application
├── test.py               # Testing utilities
├── sleepnet.pth          # Trained model weights
├── requirements.txt      # Python dependencies
├── realish_upload.csv    # Raw EEG data
├── realish_upload.meta.json  # Data metadata
└── data/
    └── processed/        # Processed features (X.npy, y.npy)
```

## Installation

### Prerequisites

- Python 3.8+
- GPU support (optional, defaults to CPU)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/sleepnet-eeg-classifier.git
cd sleepnet-eeg-classifier
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training a New Model

```bash
python train.py
```

This will:

- Load processed EEG data from `data/processed/`
- Train the SleepNet model with class balancing
- Save the trained model to `sleepnet.pth`
- Display validation metrics during training

### Evaluating a Model

```bash
python evaluate.py
```

Generates classification reports, F1-scores, and performance metrics on the test set.

### Running the Web Interface

```bash
python app.py
```

The Flask application will start on `http://localhost:5000`. You can:

- Upload EEG data files
- Get real-time sleep stage predictions
- Visualize classification results

### Testing

```bash
python test.py
```

Runs unit tests to verify model functionality.

### Preprocessing Raw Data

```bash
python preprocess.py
```

Converts raw EEG data into processed features saved as numpy arrays in `data/processed/`.

## Data Format

### Input

- **EEG Signal**: Time-series data sampled at 100 Hz
- **Epoch Duration**: 30-second windows (3000 samples per epoch)
- **Preprocessing**: Bandpass filter (0.5-30 Hz) + z-score normalization

### Output

- **Classification**: One of five sleep stages per epoch
- **Confidence**: Softmax probabilities for each stage

## Model Architecture

### SleepNet Components

1. **CNN Feature Extractor**
   - Input: 1x3000 (raw EEG)
   - Conv1d layers with increasing channels
   - Progressive downsampling

2. **Temporal Convolutional Network (TCN)**
   - Dilated convolutions for temporal context
   - Residual connections
   - Batch normalization and dropout

3. **Attention Mechanism**
   - Learns to focus on important temporal regions
   - Weighted sum of features
   - Outputs fixed-size representation

4. **Classification Head**
   - Fully connected layers
   - Dropout regularization
   - 5-way softmax for sleep stage prediction

## Configuration

Edit key parameters in `train.py`:

- `BATCH_SIZE`: Training batch size (default: 64)
- `EPOCHS`: Number of training epochs (default: 50)
- `LEARNING_RATE`: Learning rate (default: 1e-3)
- `PATIENCE`: Early stopping patience (default: 10)

## Performance

The model achieves high accuracy on multi-class sleep stage classification with balanced F1-scores across all stages.

## Requirements

See `requirements.txt` for complete dependencies:

- `torch` - PyTorch deep learning framework
- `numpy` - Numerical computations
- `mne` - EEG signal processing
- `scikit-learn` - Machine learning utilities
- `matplotlib` - Visualization
- `seaborn` - Statistical visualizations
- `tqdm` - Progress bars
- `flask` - Web framework

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Citation

If you use SleepNet in your research, please cite:

```
[Add citation information if applicable]
```

## Contact

For questions or support, please open an issue on GitHub.
