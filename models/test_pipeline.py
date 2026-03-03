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

# load checkpoint to infer architecture parameters
state = torch.load(CLASSIFIER_MODEL_PATH, map_location=device)
# infer embedding size from saved weights
embed_dim = state['embedding.weight'].shape[1]
# hidden_dim can be obtained from weight_hh_l0 shape: (4*hidden_dim, hidden_dim)
hidden_dim = state['lstm.weight_hh_l0'].shape[1]
# number of layers = count of weight_ih_lN keys without "reverse"
num_layers = len([k for k in state.keys() if k.startswith('lstm.weight_ih_l') and 'reverse' not in k])
# bidirectional if any reverse weights exist
bidirectional = any('reverse' in k for k in state.keys())

cipher_classifier = CipherClassifier(VOCAB_SIZE,
                                    embed_dim=embed_dim,
                                    hidden_dim=hidden_dim,
                                    num_layers=num_layers,
                                    bidirectional=bidirectional)
# now load parameters
cipher_classifier.load_state_dict(state)
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


# helper decryptions used in brute force sections
from itertools import permutations, product


def brute_force_caesar(ciphertext):
    letters = string.ascii_lowercase
    best_shift = None
    best_score = -1.0
    best_plain = None
    low = ciphertext.lower()
    for shift in range(1, 26):
        out = []
        for ch in low:
            if ch in letters:
                out.append(chr((ord(ch) - 97 - shift) % 26 + 97))
            else:
                out.append(ch)
        cand = "".join(out)
        sc = score_plaintext(cand)
        if sc > best_score:
            best_score = sc
            best_shift = shift
            best_plain = cand
    return best_shift, best_score, best_plain


def brute_force_vigenere(ciphertext, max_len=4):
    letters = string.ascii_lowercase
    best_key = None
    best_score = -1.0
    best_plain = None
    for length in range(1, max_len+1):
        for key_tuple in product(letters, repeat=length):
            key = "".join(key_tuple)
            plain = load_decoder_module("vigenere").decrypt(ciphertext, key)
            sc = score_plaintext(plain)
            if sc > best_score:
                best_score = sc
                best_key = key
                best_plain = plain
    return best_key, best_score, best_plain


def brute_force_skip(ciphertext, max_key=50):
    best_key = None
    best_score = -1.0
    best_plain = None
    for key in range(1, min(max_key, len(ciphertext))):
        plain = load_decoder_module("skip").decrypt(ciphertext, key)
        sc = score_plaintext(plain)
        if sc > best_score:
            best_score = sc
            best_key = key
            best_plain = plain
    return best_key, best_score, best_plain


def brute_force_columnar(ciphertext, max_cols=6):
    best_key = None
    best_score = -1.0
    best_plain = None
    for ncols in range(3, max_cols+1):
        for perm in permutations(range(ncols)):
            key = ''.join(str(p+1) for p in perm)
            plain = load_decoder_module("columnar").decrypt(ciphertext, key)
            sc = score_plaintext(plain)
            if sc > best_score:
                best_score = sc
                best_key = key
                best_plain = plain
    return best_key, best_score, best_plain



def score_plaintext(text):
    words = set(w.strip(" .,!?'" ) for w in text.split())
    score = sum(1 for w in words if w in COMMON_WORDS)
    letters = string.ascii_lowercase
    letter_frac = sum(1 for c in text.lower() if c in letters or c==' ') / max(1, len(text))
    return score + 0.5 * letter_frac

def caesar_fallback_detect(ciphertext):
    """Return (is_caesar, best_shift, best_score).

    The score is the same common-word + letter-fraction metric used elsewhere.
    """
    best_shift = None
    best_score = -1.0
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
        words = set(w.strip(" .,!?'" ) for w in candidate.split())
        score = sum(1 for w in words if w in COMMON_WORDS)
        letter_frac = sum(1 for c in candidate if c in letters or c==' ') / max(1, len(candidate))
        score = score + 0.5 * letter_frac
        if score > best_score:
            best_score = score
            best_shift = shift
    return (best_score >= 1.0), best_shift, best_score



# threshold below which we trust a simple Caesar check
DEFAULT_CONF_THRESHOLD = 0.60

# minimum quality score below which we try alternate ciphers
DEFAULT_PLAINTEXT_SCORE_THRESHOLD = 1.5

def decode_with_models(ciphertext, conf_threshold=DEFAULT_CONF_THRESHOLD, brute_force=False):
    """Decode with classifier and optional brute-force fallback.

    When ``brute_force`` is True the function exhaustively searches small key
    spaces for all ciphers, which guarantees correct output at the cost of CPU.
    """
    predicted_cipher, conf = predict_cipher(ciphertext)
    predicted_key = None
    plaintext = None

    # always run Caesar heuristic to catch confident misclassifications
    is_c, fallback_shift, caesar_score_val = caesar_fallback_detect(ciphertext)
    if is_c:
        # if classifier disagrees or is below threshold, override
        if predicted_cipher != "caesar" or conf < conf_threshold or caesar_score_val > 2.0:
            print(f"[fallback] Caesar heuristic ({caesar_score_val:.2f}) overrides classifier ({predicted_cipher}, conf={conf:.2f})")
            predicted_cipher = "caesar"
            predicted_key = fallback_shift
            plaintext = load_decoder_module("caesar").decrypt(ciphertext, predicted_key)
            return {
                "cipher": predicted_cipher,
                "key": predicted_key,
                "plaintext": plaintext,
                "classifier_confidence": float(conf)
            }

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
        for perm in permutations(range(ncols)):
            decoder = load_decoder_module("columnar")
            trial = decoder.decrypt(ciphertext, ''.join(str(p+1) for p in perm))
            score = score_plaintext(trial)
            if score > best_score:
                best_score = score
                best_plain = trial
                best_key = ''.join(str(p+1) for p in perm)
        predicted_key = best_key
        plaintext = best_plain

    if plaintext is None:
        decoder = load_decoder_module(predicted_cipher)
        plaintext = decoder.decrypt(ciphertext, predicted_key)

    # if the decrypted text looks poor, try all other ciphers and pick the best
    base_score = score_plaintext(plaintext)
    if brute_force or base_score < DEFAULT_PLAINTEXT_SCORE_THRESHOLD:
        if base_score < DEFAULT_PLAINTEXT_SCORE_THRESHOLD:
            print(f"[fallback] low plaintext score {base_score:.2f}, rescoring all ciphers")
        best = (base_score, predicted_cipher, predicted_key, plaintext)
        # first try simple re-prediction keys
        for cipher in CIPHERS:
            if cipher == predicted_cipher:
                continue
            if cipher == "caesar":
                key = predict_key_caesar(ciphertext)
                plain = load_decoder_module("caesar").decrypt(ciphertext, key)
            elif cipher == "vigenere":
                key = predict_key_vigenere(ciphertext)
                plain = load_decoder_module("vigenere").decrypt(ciphertext, key)
            elif cipher == "skip":
                key = predict_key_skip(ciphertext)
                plain = load_decoder_module("skip").decrypt(ciphertext, key)
            elif cipher == "columnar":
                ncols = predict_key_columnar(ciphertext)
                best_score2 = -1
                best_plain2 = None
                best_key2 = None
                for perm in permutations(range(ncols)):
                    trial = load_decoder_module("columnar").decrypt(ciphertext, ''.join(str(p+1) for p in perm))
                    sc = score_plaintext(trial)
                    if sc > best_score2:
                        best_score2 = sc
                        best_plain2 = trial
                        best_key2 = ''.join(str(p+1) for p in perm)
                key = best_key2
                plain = best_plain2
            sc = score_plaintext(plain)
            if sc > best[0]:
                best = (sc, cipher, key, plain)
        # second, if allowed, brute-force each cipher's keyspace
        if brute_force:
            # Caesar brute
            shift, sc, plain = brute_force_caesar(ciphertext)
            if sc > best[0]:
                best = (sc, "caesar", shift, plain)
            # Vigenere brute
            vk, sc, plain = brute_force_vigenere(ciphertext)
            if sc > best[0]:
                best = (sc, "vigenere", vk, plain)
            # Skip brute
            sk, sc, plain = brute_force_skip(ciphertext)
            if sc > best[0]:
                best = (sc, "skip", sk, plain)
            # Columnar brute
            ck, sc, plain = brute_force_columnar(ciphertext)
            if sc > best[0]:
                best = (sc, "columnar", ck, plain)
        # adopt the better candidate
        _, predicted_cipher, predicted_key, plaintext = best

    return {
        "cipher": predicted_cipher,
        "key": predicted_key,
        "plaintext": plaintext,
        "classifier_confidence": float(conf)
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Decode a ciphertext using the model and fallbacks")
    parser.add_argument("ciphertext", nargs="?", default="OBKRUOXOGHULBSOLIFBBWFLRVQQPRNGKSSOTWTQSJQSSEKZZWATJKLUDIAWINFBNYPVTTMZFPKWGDKZXTJCDIGKUHUAUEKCAR",
                        help="ciphertext string to decode")
    parser.add_argument("--bruteforce", action="store_true",
                        help="perform exhaustive key search for each cipher")
    args = parser.parse_args()
    result = decode_with_models(args.ciphertext, brute_force=args.bruteforce)
    print("Ciphertext:", args.ciphertext)
    print("Predicted cipher:", result["cipher"], f"(confidence={result['classifier_confidence']:.3f})")
    print("Predicted key:", result["key"])
    print("Decoded plaintext:", result["plaintext"])
