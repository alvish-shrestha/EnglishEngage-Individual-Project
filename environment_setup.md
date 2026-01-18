# EngageEnglish Setup Guide

## Prerequisites
- Python 3.8 or higher
- A working microphone
- Internet connection (for initial setup)

## Installation

1. **Create a Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**
   Run the following command to install all necessary libraries:
   ```bash
   pip install pygame speechrecognition torch transformers pyaudio
   ```
   *Note: `pyaudio` might require additional system tools (like `portaudio`) on some systems (e.g., `brew install portaudio` on Mac).*

3. **Verify Models**
   Ensure the following directories exist in your project root or `experiments/` folder:
   - `experiments/bert_difficulty_model`
   - `experiments/t5_sentence_generator`

## Running the Game

1. **Start the Application**
   ```bash
   python3 main.py
   ```

2. **How to Play**
   - **Start**: Click the "Start Game" button.
   - **Speak**: Press and hold **SPACE** to record your voice. Read the sentence displayed on the screen.
   - **Score**: You get points for accuracy.
   - **Speed**: Answer quickly before the timer runs out!
   - **Complete**: Finish all questions to see your final score.
