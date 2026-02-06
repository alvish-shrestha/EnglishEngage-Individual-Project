import pygame
import speech_recognition as sr
import torch
from transformers import BertTokenizer, BertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
import json
import os
import math
import random
import time
import threading
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- CONFIG ------------------------

BERT_MODEL_PATH = "experiments/bert_difficulty_model"
T5_MODEL_PATH = "experiments/t5_sentence_generator"
PROGRESS_FILE = "user_progress.json"

WIDTH, HEIGHT = 1000, 700
FPS = 120

# --- Colors (Premium Palette) ---
WHITE = (250, 250, 250)
BLACK = (20, 20, 20)
BG_GRADIENT_TOP = (30, 41, 59)
BG_GRADIENT_BOTTOM = (15, 23, 42)
ACCENT_PRIMARY = (56, 189, 248)
ACCENT_SECONDARY = (129, 140, 248)
SUCCESS_COLOR = (74, 222, 128)
ERROR_COLOR = (248, 113, 113)
TEXT_PRIMARY = (241, 245, 249)
TEXT_SECONDARY = (148, 163, 184)
SHADOW_COLOR = (0, 0, 0, 150)
PANEL_BG = (255, 255, 255, 20)

# --- Game Settings ---
MAX_ROUNDS = 10
TIME_LIMIT_PER_ROUND = 15
DATA_FILE = "data/cleaned_sentences.txt"

# --- Dynamic Content ---
POSITIVE_FEEDBACK = [
    "Excellent!", "Great Job!", "Perfect!", 
    "Amazing!", "Spot on!", "You got it!",
    "Fantastic!", "Keep it up!"
]

# ------------------ LOAD MODELS -------------------

print("Loading models... please wait")

# Load Sentences from File
def load_sentences():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if lines:
            return lines
    return ["The sun rises in the east.", "I love learning python.", "Artificial Intelligence is the future."]

SENTENCE_POOL = load_sentences()

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

try:
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH).to(device)
    bert_model.eval()

    t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_PATH)
    t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_PATH).to(device)
    t5_model.eval()
    MODELS_LOADED = True
    print(f"Models loaded on device: {device}")
except Exception as e:
    print(f"Error loading models: {e}")
    MODELS_LOADED = False
    # Fallback for UI testing without models
    device = "cpu"

# -------------- Speech recognition setup -----------

recognizer = sr.Recognizer()
mic = sr.Microphone()

# ----------------- Utility functions ----------------

def load_progress():
    default_progress = {"total_points": 0, "games_played": 0}
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                data = json.load(f)
                # Ensure keys exist (migration from old format)
                if "total_points" not in data:
                    data["total_points"] = data.get("points", 0)
                if "games_played" not in data:
                    data["games_played"] = data.get("attempts", 0)
                return data
        except Exception as e:
            print(f"Error loading progress: {e}")
            return default_progress
    return default_progress

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)

def predict_difficulty(user_response, current_sentence):
    if not MODELS_LOADED:
        return random.random()
    
    inputs = bert_tokenizer.encode_plus(user_response, current_sentence, return_tensors='pt', truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
    probs = torch.softmax(outputs.logits, dim=1)
    score = probs[0][1].item() 
    return score

def get_random_sentence():
    return random.choice(SENTENCE_POOL)

def generate_next_sentence(difficulty_score):
    return get_random_sentence()

def get_speech_input():
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            # Listen with a timeout
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=15)
        result = recognizer.recognize_google(audio)
        return result.lower()
    except sr.WaitTimeoutError:
        return None
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None
    except Exception as e:
        print(f"Mic error: {e}")
        return None

# ----------------- UI Helper Classes ----------------

def draw_gradient(surface, top_color, bottom_color):
    """Vertical gradient."""
    height = surface.get_height()
    for y in range(height):
        alpha = y / height
        r = int(top_color[0] * (1 - alpha) + bottom_color[0] * alpha)
        g = int(top_color[1] * (1 - alpha) + bottom_color[1] * alpha)
        b = int(top_color[2] * (1 - alpha) + bottom_color[2] * alpha)
        pygame.draw.line(surface, (r, g, b), (0, y), (surface.get_width(), y))

def draw_rounded_rect(surface, rect, color, radius=15):
    """Draws a rounded rectangle."""
    rect = pygame.Rect(rect)
    color = pygame.Color(*color)
    alpha = color.a
    color.a = 0
    pos = rect.topleft
    rect.topleft = 0,0
    rectangle = pygame.Surface(rect.size,pygame.SRCALPHA)
    
    circle = pygame.Surface([min(rect.size)*3]*2,pygame.SRCALPHA)
    pygame.draw.ellipse(circle,(0,0,0),circle.get_rect(),0)
    circle = pygame.transform.smoothscale(circle,[int(min(rect.size)*0.5)]*2)

    radius = rectangle.blit(circle,(0,0))
    radius.bottomright = rect.bottomright
    rectangle.blit(circle,radius)
    radius.topright = rect.topright
    rectangle.blit(circle,radius)
    radius.bottomleft = rect.bottomleft
    rectangle.blit(circle,radius)

    rectangle.fill((0,0,0),rect.inflate(-radius.w,0))
    rectangle.fill((0,0,0),rect.inflate(0,-radius.h))

    rectangle.fill(color,special_flags=pygame.BLEND_RGBA_MAX)
    rectangle.fill((255,255,255,alpha),special_flags=pygame.BLEND_RGBA_MIN)

    surface.blit(rectangle,pos)

class Button:
    def __init__(self, text, x, y, w, h, color, hover_color, text_color, font):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font = font
        self.is_hovered = False

    def draw(self, screen):
        color = self.hover_color if self.is_hovered else self.color
        draw_rounded_rect(screen, self.rect, color, radius=12)
        
        # Shadow/Border
        pygame.draw.rect(screen, (255,255,255, 30), self.rect, width=2, border_radius=12)

        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.is_hovered:
            return True
        return False

# ----------------- Game Class ----------------

class EngageEnglishGame:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("EngageEnglish: Speed & Fluency")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_title = pygame.font.SysFont("Arial Rounded MT Bold", 60)
        self.font_large = pygame.font.SysFont("Helvetica", 40) # Fallback fonts
        self.font_medium = pygame.font.SysFont("Helvetica", 28)
        self.font_small = pygame.font.SysFont("Helvetica", 20)
        
        # Load progress
        self.global_progress = load_progress()
        
        # Game State
        self.state = "MENU" # MENU, PLAYING, GAMEOVER
        
        # Session State
        self.current_round = 0
        self.score = 0
        self.current_sentence = "Welcome to Engage English."
        self.timer = TIME_LIMIT_PER_ROUND
        self.listening = False
        self.feedback = ""
        self.feedback_color = WHITE
        
        # Buttons
        self.btn_start = Button("Start Journey", WIDTH//2 - 100, HEIGHT//2 + 50, 200, 60, ACCENT_PRIMARY, ACCENT_SECONDARY, WHITE, self.font_medium)
        self.btn_retry = Button("Play Again", WIDTH//2 - 100, HEIGHT//2 + 100, 200, 60, ACCENT_PRIMARY, ACCENT_SECONDARY, WHITE, self.font_medium)

        # Pulse animation
        self.pulse = 0
        
    def reset_session(self):
        self.current_round = 1
        self.score = 0
        self.current_sentence = get_random_sentence()
        self.timer = TIME_LIMIT_PER_ROUND
        self.listening = False
        self.feedback = ""
        self.feedback_color = TEXT_SECONDARY
        self.state = "PLAYING"

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            mouse_pos = pygame.mouse.get_pos()
            
            if self.state == "MENU":
                self.btn_start.check_hover(mouse_pos)
                if self.btn_start.is_clicked(event):
                    self.reset_session()
                    
            elif self.state == "GAMEOVER":
                self.btn_retry.check_hover(mouse_pos)
                if self.btn_retry.is_clicked(event):
                    self.reset_session()
            
            elif self.state == "PLAYING":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and not self.listening:
                        self.start_listening()
                        
        return True

    def start_listening(self):
        self.listening = True
        # Redraw immediately to show listening state
        self.draw()
        pygame.display.flip()
        
        if not MODELS_LOADED:
             # Fast path for testing without models
             time.sleep(1)
             user_text = "I am excited to learn" 
        else:
            user_text = get_speech_input()
        
        self.process_attempt(user_text)
        self.listening = False

    def process_attempt(self, user_text):
        if user_text:
            difficulty = predict_difficulty(user_text, self.current_sentence)
            
            similarity = self.text_similarity(user_text, self.current_sentence)
            
            if similarity > 0.8: # Threshold for correct
                self.score += int(100 * (self.timer / TIME_LIMIT_PER_ROUND)) + 50
                self.feedback = random.choice(POSITIVE_FEEDBACK)
                self.feedback_color = SUCCESS_COLOR
                # Increase difficulty (conceptually, by picking next sentence)
                self.current_sentence = get_random_sentence() 
            else:
                self.feedback = f"Heard: '{user_text}'"
                self.feedback_color = ERROR_COLOR
                # Try another sentence
                self.current_sentence = get_random_sentence()
            
            self.current_round += 1
            self.timer = TIME_LIMIT_PER_ROUND
            
            if self.current_round > MAX_ROUNDS:
                self.end_game()
        else:
            self.feedback = "Couldn't hear you. Try again!"
            self.feedback_color = ERROR_COLOR

    def text_similarity(self, s1, s2):
        import string
        # Normalize: lower case and remove all punctuation
        translator = str.maketrans('', '', string.punctuation)
        s1 = s1.lower().translate(translator).strip()
        s2 = s2.lower().translate(translator).strip()
        return 1.0 if s1 == s2 else 0.0

    def end_game(self):
        self.state = "GAMEOVER"
        self.global_progress["total_points"] += self.score
        self.global_progress["games_played"] += 1
        save_progress(self.global_progress)

    def update(self):
        self.pulse += 0.05
        
        if self.state == "PLAYING":
            if not self.listening:
                self.timer -= 1 / FPS
                if self.timer <= 0:
                    self.feedback = "Time's up!"
                    self.feedback_color = ERROR_COLOR
                    self.current_round += 1
                    self.timer = TIME_LIMIT_PER_ROUND
                    self.current_sentence = get_random_sentence()
                    
                    if self.current_round > MAX_ROUNDS:
                        self.end_game()

    def draw(self):
        # Background
        draw_gradient(self.screen, BG_GRADIENT_TOP, BG_GRADIENT_BOTTOM)
        
        if self.state == "MENU":
            self.draw_menu()
        elif self.state == "PLAYING":
            self.draw_playing()
        elif self.state == "GAMEOVER":
            self.draw_gameover()
            
        pygame.display.flip()

    def draw_menu(self):
        title = self.font_title.render("EngageEnglish", True, ACCENT_PRIMARY)
        subtitle = self.font_medium.render("Master Fluency & Speed", True, TEXT_SECONDARY)
        
        self.screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//3))
        self.screen.blit(subtitle, (WIDTH//2 - subtitle.get_width()//2, HEIGHT//3 + 80))
        
        self.btn_start.draw(self.screen)
        
        # Stats
        stats = f"Total Points: {self.global_progress['total_points']}"
        stats_surf = self.font_small.render(stats, True, TEXT_SECONDARY)
        self.screen.blit(stats_surf, (WIDTH//2 - stats_surf.get_width()//2, HEIGHT - 50))

    def draw_playing(self):
        # Header
        header_rect = pygame.Rect(20, 20, WIDTH-40, 60)
        draw_rounded_rect(self.screen, header_rect, (255, 255, 255, 30))
        
        score_text = self.font_medium.render(f"Score: {self.score}", True, ACCENT_PRIMARY)
        round_text = self.font_medium.render(f"Round: {self.current_round}/{MAX_ROUNDS}", True, TEXT_PRIMARY)
        
        self.screen.blit(score_text, (40, 35))
        self.screen.blit(round_text, (WIDTH - 40 - round_text.get_width(), 35))
        
        # Timer Bar
        timer_width = (self.timer / TIME_LIMIT_PER_ROUND) * (WIDTH - 40)
        timer_rect = pygame.Rect(20, 90, timer_width, 5)
        timer_color = SUCCESS_COLOR if self.timer > 5 else ERROR_COLOR
        pygame.draw.rect(self.screen, timer_color, timer_rect, border_radius=5)
        
        # Content Panel
        panel_rect = pygame.Rect(100, 150, WIDTH-200, 300)
        draw_rounded_rect(self.screen, panel_rect, PANEL_BG, radius=20)
        
        # Sentence
        label = self.font_small.render("READ ALOUD:", True, TEXT_SECONDARY)
        self.screen.blit(label, (WIDTH//2 - label.get_width()//2, 180))
        
        sent_surf = self.font_large.render(f'"{self.current_sentence}"', True, WHITE)
        # Handle simple wrapping if needed roughly (center it)
        rect = sent_surf.get_rect(center=(WIDTH//2, 280))
        self.screen.blit(sent_surf, rect)
        
        # Feedback
        feed_surf = self.font_medium.render(self.feedback, True, self.feedback_color)
        feed_rect = feed_surf.get_rect(center=(WIDTH//2, 380))
        self.screen.blit(feed_surf, feed_rect)
        
        # Mic Indicator
        mic_y = 550
        scale = 1.0 + 0.1 * math.sin(self.pulse * 5)
        color = ERROR_COLOR if self.listening else ACCENT_PRIMARY
        radius = int(40 * scale) if self.listening else 40
        
        pygame.draw.circle(self.screen, color, (WIDTH//2, mic_y), radius)
        
        icon_text = "..." if self.listening else "MIC"
        icon = self.font_small.render(icon_text, True, BLACK)
        icon_rect = icon.get_rect(center=(WIDTH//2, mic_y))
        self.screen.blit(icon, icon_rect)
        
        msg = "Hold SPACE to Speak" if not self.listening else "Listening..."
        hint = self.font_small.render(msg, True, TEXT_SECONDARY)
        self.screen.blit(hint, (WIDTH//2 - hint.get_width()//2, mic_y + 60))

    def draw_gameover(self):
        title = self.font_title.render("Session Complete!", True, SUCCESS_COLOR)
        score_surf = self.font_large.render(f"Final Score: {self.score}", True, WHITE)
        
        self.screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//3 - 50))
        self.screen.blit(score_surf, (WIDTH//2 - score_surf.get_width()//2, HEIGHT//3 + 50))
        
        self.btn_retry.draw(self.screen)

    def run(self):
        running = True
        while running:
            running = self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()

if __name__ == "__main__":
    game = EngageEnglishGame()
    game.run()
