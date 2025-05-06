
---

## 🧠 Grammar Scoring System

A machine learning pipeline for automatic grammar scoring using audio input. It leverages **Wav2Vec2** for speech-to-text transcription, **Librosa** for audio feature extraction, and NLP techniques for textual analysis. The final grammar score is predicted using regression models trained on combined features.

---

### 📁 Project Structure

```
Grammar_Scoring/
├── Dataset/
│   ├── audios/
│   │   ├── train/
│   │   └── test/
│   ├── train.csv
│   └── test.csv
├── Grammar_Scoring.ipynb
├── README.md
```

---

### 🚀 Features

* 🎙️ **ASR**: Uses pre-trained Wav2Vec2 model for accurate transcription.
* 🎧 **Audio Features**: MFCCs, Zero Crossing Rate, Spectral Centroid, etc. via Librosa.
* 📖 **Text Features**: Grammar-checking metrics, sentence length, part-of-speech distributions, etc.
* 🔍 **Model**: Trained regression model to predict grammar quality on a scale of 0 to 5.
* 📈 **Evaluation**: Reports MAE, MSE, RMSE, and R² for performance benchmarking.

---

### ⚙️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

**Key Libraries:**

* `transformers`
* `librosa`
* `scikit-learn`
* `pandas`, `numpy`
* `torch`
* `soundfile`, `jiwer` (for WER metrics if needed)

---

### 🛠️ How It Works

1. **Input Audio** → Transcribed via Wav2Vec2
2. **Extract Features**:

   * Audio features using Librosa
   * Textual features (grammar rules, POS tags, etc.)
3. **Model Prediction**: Regression model outputs grammar score.
4. **Evaluation**: Metrics computed to assess model performance.

---

### 📊 Sample Metrics Output

```bash
📊 Model Evaluation Metrics:
Mean Absolute Error (MAE): 0.34
Mean Squared Error (MSE): 0.21
Root Mean Squared Error (RMSE): 0.46
R² Score: 0.87
```

---

### 🧪 Example Use

```python
# Load audio and extract prediction
pred = model.predict(extracted_features)
score = int(np.clip(pred[0], 0, 5))
print(f"Predicted grammar score: {score}")
```

---

---

### 🧑‍💻 Author

Bhumik Kumar Kapoor
[LinkedIn](https://linkedin.com/in/bhumik-kumar-kapoor-02920b2a0) | [GitHub](https://github.com/kbhumik27)

---


