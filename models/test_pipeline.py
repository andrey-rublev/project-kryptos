import os
import sys
import importlib.util
import torch
import string

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.dirname(__file__)
CLASSIFIER_MODEL_PATH = os.path.join(MODELS_DIR, "cipher_classifier.pt")
CAESAR_KEY_MODEL_PATH = os.path.join(MODELS_DIR, "key_predictor_caesar.pt")

sys.path.insert(0, MODELS_DIR)

from cipher_classifier import CipherClassifier, encode_text, VOCAB_SIZE, CIPHERS
from key_predictor_caesar_model import CaesarKeyPredictor
# additional key model classes
from key_predictor_vigenere_model import VigenereKeyPredictor
from key_predictor_skip_model import SkipKeyPredictor
from key_predictor_columnar_model import ColumnarKeyPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cipher_classifier = CipherClassifier(VOCAB_SIZE)
cipher_classifier.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device))
cipher_classifier.to(device)
cipher_classifier.eval()

caesar_key_model = CaesarKeyPredictor()
state = torch.load(CAESAR_KEY_MODEL_PATH, map_location=device)
try:
    caesar_key_model.load_state_dict(state)
except Exception:
    caesar_key_model.load_state_dict(state, strict=False)
caesar_key_model.to(device)
caesar_key_model.eval()

# load other key predictors if available
vigenere_key_model = VigenereKeyPredictor()
vig_path = os.path.join(MODELS_DIR, "key_predictor_vigenere.pt")
if os.path.exists(vig_path):
    vstate = torch.load(vig_path, map_location=device)
    try:
        vigenere_key_model.load_state_dict(vstate)
    except Exception:
        vigenere_key_model.load_state_dict(vstate, strict=False)
    vigenere_key_model.to(device)
    vigenere_key_model.eval()

skip_key_model = SkipKeyPredictor()
sk_path = os.path.join(MODELS_DIR, "key_predictor_skip.pt")
if os.path.exists(sk_path):
    sstate = torch.load(sk_path, map_location=device)
    try:
        skip_key_model.load_state_dict(sstate)
    except Exception:
        skip_key_model.load_state_dict(sstate, strict=False)
    skip_key_model.to(device)
    skip_key_model.eval()

columnar_key_model = ColumnarKeyPredictor()
col_path = os.path.join(MODELS_DIR, "key_predictor_columnar.pt")
if os.path.exists(col_path):
    colstate = torch.load(col_path, map_location=device)
    try:
        columnar_key_model.load_state_dict(colstate)
    except Exception:
        columnar_key_model.load_state_dict(colstate, strict=False)
    columnar_key_model.to(device)
    columnar_key_model.eval()

ALL_CHARS = string.ascii_lowercase + " .,!?'"
CHAR2IDX = {c: i for i, c in enumerate(ALL_CHARS)}
# key prediction uses same char set but shifted by 1, reserve 0 for padding
KEY_CHARS = ALL_CHARS
KEY_CHAR2IDX = {c: i+1 for i, c in enumerate(KEY_CHARS)}
KEY_IDX2CHAR = {v: k for k, v in KEY_CHAR2IDX.items()}
MAX_LEN = 200

def encode_text_key(text):
    text = text.lower()[:MAX_LEN]
    idxs = [CHAR2IDX.get(c, 0) for c in text]
    if len(idxs) < MAX_LEN:
        idxs += [0] * (MAX_LEN - len(idxs))
    return idxs


def predict_cipher(text):
    with torch.no_grad():
        encoded = torch.tensor([encode_text(text)], dtype=torch.long).to(device)
        logits = cipher_classifier(encoded)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        return CIPHERS[pred], probs[0][pred].item()


def predict_key_caesar(text):
    with torch.no_grad():
        encoded = torch.tensor([encode_text_key(text)], dtype=torch.long).to(device)
        logits = caesar_key_model(encoded)
        pred = torch.argmax(logits, dim=1).item()
    return int(pred)


def predict_key_vigenere(text):
    with torch.no_grad():
        encoded = torch.tensor([encode_text_key(text)], dtype=torch.long).to(device)
        logits = vigenere_key_model(encoded)  # shape batch,max_len,vocab_size
        chars = logits.argmax(dim=2)[0].tolist()
    # convert indices back to letters (0 is padding)
    key = ''.join(KEY_IDX2CHAR.get(c, '') for c in chars if c != 0)
    return key


def predict_key_skip(text):
    with torch.no_grad():
        encoded = torch.tensor([encode_text_key(text)], dtype=torch.long).to(device)
        logits = skip_key_model(encoded)
        pred = torch.argmax(logits, dim=1).item()
    return int(pred)


def predict_key_columnar(text):
    with torch.no_grad():
        encoded = torch.tensor([encode_text_key(text)], dtype=torch.long).to(device)
        logits = columnar_key_model(encoded)
        pred = torch.argmax(logits, dim=1).item()
    # map back 0..6 -> 3..9
    return pred + 3


def load_decoder_module(cipher_name):
    decoder_path = os.path.join(BASE_DIR, "decode", f"{cipher_name}.py")
    if not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Decoder not found: {decoder_path}")
    spec = importlib.util.spec_from_file_location(f"decoder_{cipher_name}", decoder_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON_WORDS = {"the","and","to","of","a","in","is","it","you","that"}

def caesar_fallback_detect(ciphertext):
    best_shift = None
    best_score = -1
    low_text = ciphertext.lower()
    letters = string.ascii_lowercase
    for shift in range(1, 26):
        out = []
        for ch in low_text:
            if ch in letters:
                out.append(chr((ord(ch)-97-shift) % 26 + 97))
            else:
                out.append(ch)
        candidate = "".join(out)
        words = set(w.strip(" .,!?'") for w in candidate.split())
        score = sum(1 for w in words if w in COMMON_WORDS)
        letter_frac = sum(1 for c in candidate if c in letters or c==' ') / max(1, len(candidate))
        score = score + 0.5 * letter_frac
        if score > best_score:
            best_score = score
            best_shift = shift
    if best_score >= 1.0:
        return True, best_shift
    return False, None


def decode_with_models(ciphertext):
    predicted_cipher, conf = predict_cipher(ciphertext)
    predicted_key = None
    plaintext = None

    if predicted_cipher == "caesar":
        predicted_key = predict_key_caesar(ciphertext)
    elif predicted_cipher == "vigenere":
        predicted_key = predict_key_vigenere(ciphertext)
    elif predicted_cipher == "skip":
        predicted_key = predict_key_skip(ciphertext)
    elif predicted_cipher == "columnar":
        # predict column count and brute-force the order
        ncols = predict_key_columnar(ciphertext)
        best_score = -1
        best_plain = None
        best_key = None
        from itertools import permutations
        for perm in permutations(range(ncols)):
            decoder = load_decoder_module("columnar")
            trial = decoder.decrypt(ciphertext, ''.join(str(p+1) for p in perm))
            words = set(w.strip(" .,!?'" ) for w in trial.split())
            score = sum(1 for w in words if w in COMMON_WORDS)
            if score > best_score:
                best_score = score
                best_plain = trial
                best_key = ''.join(str(p+1) for p in perm)
        predicted_key = best_key
        plaintext = best_plain
    else:
        # fallback Caesar detection
        is_caesar, fallback_shift = caesar_fallback_detect(ciphertext)
        if is_caesar:
            predicted_cipher = "caesar"
            predicted_key = fallback_shift

    if plaintext is None:
        decoder = load_decoder_module(predicted_cipher)
        plaintext = decoder.decrypt(ciphertext, predicted_key)

    return {
        "cipher": predicted_cipher,
        "key": predicted_key,
        "plaintext": plaintext,
        "classifier_confidence": float(conf)
    }


if __name__ == "__main__":
    example = "yaa,mhrwi a s laauh ceyl belyitetoneb  eong  i inurftadntlato oice  ov ,end bwtgu ifason nto     dnhuoufacriuet n, soioi inr pfdacertbptc gerfrenoef  iat ttrl ncmeowe  nhdeoolhoce s  , ma gimtiahunlb"
    result = decode_with_models(example)
    print("Ciphertext:", example)
    print("Predicted cipher:", result["cipher"], f"(confidence={result['classifier_confidence']:.3f})")
    print("Predicted key:", result["key"])
    print("Decoded plaintext:", result["plaintext"])
