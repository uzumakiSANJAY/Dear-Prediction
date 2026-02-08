import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from collections import Counter
import json
import os
import hashlib
import logging
from datetime import date

logger = logging.getLogger(__name__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from xgboost import XGBRegressor

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available")

PRIZE_KEYS = ["mc", "1st", "cons", "2nd", "3rd", "4th", "5th"]

PRIZE_COUNTS = {
    "mc": 1,
    "1st": 1,
    "cons": 1,
    "2nd": 10,
    "3rd": 10,
    "4th": 10,
    "5th": 90,
}


def make_seed(time_slot: str, date_str: str = None) -> int:
    if date_str is None:
        date_str = str(date.today())
    raw = f"{date_str}:{time_slot}"
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


def extract_per_category(draws):
    category_sequences = {k: [] for k in PRIZE_KEYS}

    for row in draws:
        prizes = row["prizes"]
        if isinstance(prizes, str):
            prizes = json.loads(prizes)

        for key in PRIZE_KEYS:
            val = prizes.get(key, [])
            if isinstance(val, str):
                val = [val]
            if not isinstance(val, list):
                val = [str(val)]

            nums = []
            for v in val:
                digits = "".join(c for c in str(v) if c.isdigit())
                if digits:
                    nums.append(int(digits[-4:]) if len(digits) >= 4 else int(digits))

            if nums:
                category_sequences[key].append(nums)

    return category_sequences


# ---------------------------------------------------------------------------
# Rich feature engineering
# ---------------------------------------------------------------------------

def build_features(window_nums):
    """Build a rich feature vector from a window of numbers."""
    feats = list(window_nums)  # raw values

    # Last digit of each
    feats += [n % 10 for n in window_nums]
    # Tens digit
    feats += [(n // 10) % 10 for n in window_nums]
    # Hundreds digit
    feats += [(n // 100) % 10 for n in window_nums]
    # Thousands digit
    feats += [(n // 1000) % 10 for n in window_nums]

    # Differences between consecutive
    diffs = [window_nums[i] - window_nums[i - 1] for i in range(1, len(window_nums))]
    feats += diffs

    # Stats
    arr = np.array(window_nums, dtype=float)
    feats.append(float(np.mean(arr)))
    feats.append(float(np.std(arr)))
    feats.append(float(np.min(arr)))
    feats.append(float(np.max(arr)))
    feats.append(float(np.median(arr)))

    # Odd/even counts
    feats.append(sum(1 for n in window_nums if n % 2 == 0))
    feats.append(sum(1 for n in window_nums if n % 2 == 1))

    # Range buckets (0-2499, 2500-4999, 5000-7499, 7500-9999)
    for lo in [0, 2500, 5000, 7500]:
        feats.append(sum(1 for n in window_nums if lo <= n < lo + 2500))

    # Sum of last two digits
    feats.append(sum(n % 100 for n in window_nums))

    return feats


def build_dataset(flat, window=7):
    """Build X, y from a flat sequence of numbers with rich features."""
    if len(flat) < window + 2:
        return None, None

    X, y = [], []
    for i in range(window, len(flat)):
        feats = build_features(flat[i - window : i])
        X.append(feats)
        y.append(flat[i])

    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)


# ---------------------------------------------------------------------------
# Digit-position analyzer: predicts each digit position independently
# ---------------------------------------------------------------------------

class DigitPositionAnalyzer:
    """Analyzes and predicts each digit position (thousands, hundreds, tens, ones) independently."""

    def __init__(self):
        self.digit_models = {}  # key -> [model_d3, model_d2, model_d1, model_d0]
        self.is_trained = False

    def fit(self, category_sequences):
        window = 7

        for key in PRIZE_KEYS:
            seqs = category_sequences.get(key, [])
            flat = [s[0] for s in seqs if s]

            if len(flat) < window + 10:
                continue

            models_for_key = []

            for digit_pos in range(4):  # d3(thousands) d2(hundreds) d1(tens) d0(ones)
                X, y_digit = [], []
                for i in range(window, len(flat)):
                    feats = build_features(flat[i - window : i])
                    target_num = flat[i]
                    digit_val = (target_num // (10 ** digit_pos)) % 10
                    X.append(feats)
                    y_digit.append(digit_val)

                X = np.array(X, dtype=np.float64)
                y_digit = np.array(y_digit, dtype=np.float64)

                model = GradientBoostingRegressor(
                    n_estimators=300, max_depth=5, learning_rate=0.05,
                    subsample=0.8, random_state=42
                )
                model.fit(X, y_digit)
                models_for_key.append(model)

            self.digit_models[key] = {
                "models": models_for_key,
                "last_window": flat[-window:],
            }

        self.is_trained = bool(self.digit_models)
        if self.is_trained:
            logger.info(f"DigitPosition trained for {list(self.digit_models.keys())}")

    def predict_number(self, key):
        if key not in self.digit_models:
            return None
        m = self.digit_models[key]
        feats = np.array(build_features(m["last_window"])).reshape(1, -1)

        digits = []
        for model in m["models"]:
            pred = model.predict(feats)[0]
            digit = int(round(np.clip(pred, 0, 9)))
            digits.append(digit)

        # digits = [d0(ones), d1(tens), d2(hundreds), d3(thousands)]
        number = digits[0] + digits[1] * 10 + digits[2] * 100 + digits[3] * 1000
        return number % 10000


# ---------------------------------------------------------------------------
# Statistical analyzer
# ---------------------------------------------------------------------------

class StatisticalAnalyzer:
    def __init__(self):
        self.category_freq = {}
        self.category_recent_freq = {}
        self.hot = {}
        self.cold = {}
        self.last_digit_freq = Counter()
        self.pair_freq = Counter()
        self.digit_pos_freq = {}  # key -> [Counter for d3, d2, d1, d0]

    def fit(self, category_sequences):
        for key in PRIZE_KEYS:
            seqs = category_sequences.get(key, [])
            all_nums = [n for seq in seqs for n in seq]
            recent_nums = [n for seq in seqs[-30:] for n in seq]

            self.category_freq[key] = Counter(all_nums)
            self.category_recent_freq[key] = Counter(recent_nums)

            self.hot[key] = [n for n, _ in Counter(recent_nums).most_common(15)]
            overall_set = set(all_nums)
            recent_set = set(recent_nums)
            self.cold[key] = list(overall_set - recent_set)[:15]

            # Per digit-position frequency
            pos_counters = [Counter() for _ in range(4)]
            for n in all_nums:
                for p in range(4):
                    d = (n // (10 ** p)) % 10
                    pos_counters[p][d] += 1
            self.digit_pos_freq[key] = pos_counters

        all_first = [n for seq in category_sequences.get("1st", []) for n in seq]
        self.last_digit_freq = Counter(n % 10 for n in all_first)

        for seq in category_sequences.get("1st", []):
            for n in seq:
                d1 = (n // 10) % 10
                d2 = n % 10
                self.pair_freq[(d1, d2)] += 1

    def predict_numbers(self, key, count, rng):
        freq = self.category_recent_freq.get(key, Counter())
        if not freq:
            freq = self.category_freq.get(key, Counter())
        if not freq:
            return [str(rng.randint(0, 10000)).zfill(4) for _ in range(count)]

        numbers = list(freq.keys())
        weights = np.array(list(freq.values()), dtype=np.float64)
        weights = weights / weights.sum()

        results = set()
        attempts = 0
        while len(results) < count and attempts < count * 30:
            chosen = rng.choice(numbers, p=weights)
            perturbation = rng.randint(-3, 4)
            result = (chosen + perturbation) % 10000
            results.add(result)
            attempts += 1

        while len(results) < count:
            # Use digit-position frequency to fill
            if key in self.digit_pos_freq:
                n = 0
                for p in range(4):
                    pc = self.digit_pos_freq[key][p]
                    digs = list(pc.keys())
                    ws = np.array(list(pc.values()), dtype=np.float64)
                    ws = ws / ws.sum()
                    d = rng.choice(digs, p=ws)
                    n += d * (10 ** p)
                results.add(n % 10000)
            else:
                results.add(rng.randint(0, 10000))

        return sorted([str(n).zfill(4) for n in results])[:count]

    def predict(self, rng):
        result = {}
        for key in PRIZE_KEYS:
            count = PRIZE_COUNTS[key]
            nums = self.predict_numbers(key, count, rng)
            result[key] = nums[0] if count == 1 else nums
        return result


# ---------------------------------------------------------------------------
# Random Forest + XGBoost ensemble predictor
# ---------------------------------------------------------------------------

class TreeEnsemblePredictor:
    def __init__(self):
        self.models = {}
        self.is_trained = False
        self.cv_scores = {}

    def fit(self, category_sequences):
        window = 7

        for key in PRIZE_KEYS:
            seqs = category_sequences.get(key, [])
            flat = [s[0] for s in seqs if s]

            X, y = build_dataset(flat, window)
            if X is None or len(X) < 15:
                continue

            # Random Forest
            rf = RandomForestRegressor(
                n_estimators=500, max_depth=20, min_samples_leaf=2,
                random_state=42, n_jobs=-1
            )
            rf.fit(X, y)

            # Gradient Boosting
            gb = GradientBoostingRegressor(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, random_state=42
            )
            gb.fit(X, y)

            # XGBoost if available
            xgb_model = None
            if XGB_AVAILABLE:
                xgb_model = XGBRegressor(
                    n_estimators=500, max_depth=7, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbosity=0
                )
                xgb_model.fit(X, y)

            # Cross-validation score for confidence
            cv = cross_val_score(rf, X, y, cv=min(5, len(X) // 3), scoring="r2")
            r2 = max(0, float(np.mean(cv)))
            self.cv_scores[key] = r2

            self.models[key] = {
                "rf": rf,
                "gb": gb,
                "xgb": xgb_model,
                "last_window": flat[-window:],
            }

        self.is_trained = bool(self.models)
        if self.is_trained:
            logger.info(f"TreeEnsemble trained for {list(self.models.keys())}, "
                        f"CV R2 scores: {self.cv_scores}")

    def predict_base(self, key):
        if key not in self.models:
            return None
        m = self.models[key]
        feats = np.array(build_features(m["last_window"])).reshape(1, -1)

        preds = [
            m["rf"].predict(feats)[0],
            m["gb"].predict(feats)[0],
        ]
        if m["xgb"] is not None:
            preds.append(m["xgb"].predict(feats)[0])

        return int(np.mean(preds)) % 10000

    def predict_numbers(self, key, count, rng):
        base = self.predict_base(key)
        if base is None:
            return None

        results = {base}
        spread = 30
        while len(results) < count:
            perturb = rng.randint(-spread, spread + 1)
            results.add((base + perturb) % 10000)
            if len(results) < count // 2:
                spread += 10

        return sorted([str(n).zfill(4) for n in results])[:count]

    def predict(self, rng):
        result = {}
        for key in PRIZE_KEYS:
            count = PRIZE_COUNTS[key]
            nums = self.predict_numbers(key, count, rng)
            if nums is not None:
                result[key] = nums[0] if count == 1 else nums
        return result


# ---------------------------------------------------------------------------
# LSTM predictor (deeper, more epochs, all categories)
# ---------------------------------------------------------------------------

class LSTMPredictor:
    def __init__(self):
        self.models = {}
        self.is_trained = False
        self.max_val = 10000

    def fit(self, category_sequences):
        if not TF_AVAILABLE:
            logger.warning("TF not available for LSTM")
            return

        window = 15

        for key in PRIZE_KEYS:
            seqs = category_sequences.get(key, [])
            flat = [s[0] for s in seqs if s]

            if len(flat) < window + 15:
                continue

            data = np.array(flat, dtype=np.float32) / self.max_val
            X, y = [], []
            for i in range(window, len(data)):
                X.append(data[i - window : i])
                y.append(data[i])

            X = np.array(X).reshape(-1, window, 1)
            y = np.array(y)

            model = keras.Sequential([
                keras.layers.LSTM(128, input_shape=(window, 1), return_sequences=True),
                keras.layers.Dropout(0.3),
                keras.layers.LSTM(64, return_sequences=True),
                keras.layers.Dropout(0.2),
                keras.layers.LSTM(32),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(1),
            ])

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss="mse",
            )

            early_stop = keras.callbacks.EarlyStopping(
                patience=15, restore_best_weights=True
            )

            model.fit(
                X, y,
                epochs=150,
                batch_size=16,
                verbose=0,
                validation_split=0.15,
                callbacks=[early_stop],
            )

            self.models[key] = {"model": model, "last_window": flat[-window:]}

        self.is_trained = bool(self.models)
        if self.is_trained:
            logger.info(f"LSTM trained for {list(self.models.keys())}")

    def predict_base(self, key):
        if key not in self.models:
            return None
        m = self.models[key]
        data = np.array(m["last_window"], dtype=np.float32) / self.max_val
        X = data.reshape(1, len(data), 1)
        pred = m["model"].predict(X, verbose=0)[0][0]
        return int(pred * self.max_val) % 10000

    def predict_numbers(self, key, count, rng):
        base = self.predict_base(key)
        if base is None:
            return None

        results = {base}
        while len(results) < count:
            perturb = rng.randint(-40, 41)
            results.add((base + perturb) % 10000)

        return sorted([str(n).zfill(4) for n in results])[:count]

    def predict(self, rng):
        result = {}
        for key in PRIZE_KEYS:
            nums = self.predict_numbers(key, PRIZE_COUNTS[key], rng)
            if nums is not None:
                result[key] = nums[0] if PRIZE_COUNTS[key] == 1 else nums
        return result


# ---------------------------------------------------------------------------
# Master ensemble
# ---------------------------------------------------------------------------

class EnsembleTrainer:
    def __init__(self):
        self.statistical = StatisticalAnalyzer()
        self.trees = TreeEnsemblePredictor()
        self.lstm = LSTMPredictor()
        self.digit_pos = DigitPositionAnalyzer()
        self.is_trained = False
        self.category_sequences = {}

    def train(self, draw_data):
        self.category_sequences = extract_per_category(draw_data)

        total_nums = sum(len(s) for seqs in self.category_sequences.values() for s in seqs)
        if total_nums == 0:
            logger.warning("No numbers extracted from data")
            return False

        logger.info(f"Training on {total_nums} numbers from {len(draw_data)} draws")

        self.statistical.fit(self.category_sequences)
        self.trees.fit(self.category_sequences)
        self.digit_pos.fit(self.category_sequences)
        self.lstm.fit(self.category_sequences)

        self.is_trained = True
        logger.info("All models trained successfully")
        return True

    def predict(self, time_slot="1pm", date_str=None):
        seed = make_seed(time_slot, date_str)
        rng = np.random.RandomState(seed)

        stat_pred = self.statistical.predict(rng)
        tree_pred = self.trees.predict(rng) if self.trees.is_trained else {}
        lstm_pred = self.lstm.predict(rng) if self.lstm.is_trained else {}

        # Digit-position predictions (single numbers only)
        dpos_pred = {}
        if self.digit_pos.is_trained:
            for key in PRIZE_KEYS:
                if PRIZE_COUNTS[key] == 1:
                    p = self.digit_pos.predict_number(key)
                    if p is not None:
                        dpos_pred[key] = str(p).zfill(4)

        final = {}
        confidence_parts = []

        for key in PRIZE_KEYS:
            count = PRIZE_COUNTS[key]

            if count == 1:
                candidates = []
                sources = []

                if key in stat_pred:
                    candidates.append(int(stat_pred[key]))
                    sources.append("stat")
                if key in tree_pred:
                    candidates.append(int(tree_pred[key]))
                    sources.append("tree")
                if key in lstm_pred:
                    candidates.append(int(lstm_pred[key]))
                    sources.append("lstm")
                if key in dpos_pred:
                    candidates.append(int(dpos_pred[key]))
                    sources.append("dpos")

                if candidates:
                    # Weighted: trees and digit-position get highest weight
                    n = len(candidates)
                    weights = []
                    for s in sources:
                        if s == "tree":
                            weights.append(0.35)
                        elif s == "dpos":
                            weights.append(0.30)
                        elif s == "lstm":
                            weights.append(0.25)
                        else:
                            weights.append(0.10)
                    wsum = sum(weights)
                    weights = [w / wsum for w in weights]

                    avg = int(np.average(candidates, weights=weights)) % 10000
                    final[key] = str(avg).zfill(4)

                    # Confidence from model count + agreement + CV score
                    model_bonus = min(0.4, n * 0.10)
                    cv_r2 = self.trees.cv_scores.get(key, 0)
                    cv_bonus = cv_r2 * 0.25

                    if n > 1:
                        spread = np.std(candidates) / 10000
                        agreement = max(0.2, 1 - spread * 1.5)
                    else:
                        agreement = 0.5

                    conf = min(0.97, agreement * 0.35 + model_bonus + cv_bonus + 0.30)
                    confidence_parts.append(conf)
                else:
                    final[key] = str(rng.randint(0, 10000)).zfill(4)
                    confidence_parts.append(0.15)
            else:
                # Multi-number: merge all sources
                all_nums = set()

                for src in [stat_pred, tree_pred, lstm_pred]:
                    if key in src and isinstance(src[key], list):
                        all_nums.update(src[key])

                # Fill from frequency-weighted sampling
                freq = self.statistical.category_recent_freq.get(key, Counter())
                if freq:
                    numbers = list(freq.keys())
                    ws = np.array(list(freq.values()), dtype=np.float64)
                    ws = ws / ws.sum()
                    while len(all_nums) < count:
                        chosen = rng.choice(numbers, p=ws)
                        perturb = rng.randint(-3, 4)
                        all_nums.add(str((chosen + perturb) % 10000).zfill(4))

                while len(all_nums) < count:
                    all_nums.add(str(rng.randint(0, 10000)).zfill(4))

                final[key] = sorted(list(all_nums))[:count]

                sources_count = sum(
                    1 for src in [stat_pred, tree_pred, lstm_pred]
                    if key in src and isinstance(src[key], list) and len(src[key]) > 0
                )
                cv_r2 = self.trees.cv_scores.get(key, 0)
                conf = min(0.95, 0.40 + sources_count * 0.12 + cv_r2 * 0.20)
                confidence_parts.append(conf)

        # Consolation = same as 1st prize
        if "1st" in final:
            final["cons"] = final["1st"]

        overall_confidence = float(np.mean(confidence_parts)) if confidence_parts else 0.15

        models_used = ["statistical"]
        if self.trees.is_trained:
            models_used.append("rf+gb+xgb")
        if self.digit_pos.is_trained:
            models_used.append("digit_position")
        if self.lstm.is_trained:
            models_used.append("lstm")

        analysis = self._build_analysis(stat_pred, tree_pred, lstm_pred, dpos_pred)

        return {
            "predicted_numbers": final,
            "confidence": round(overall_confidence, 4),
            "model_used": "+".join(models_used),
            "analysis": analysis,
        }

    def _build_analysis(self, stat_pred, tree_pred, lstm_pred, dpos_pred):
        hot = self.statistical.hot.get("1st", [])
        cold = self.statistical.cold.get("1st", [])
        ldf = dict(self.statistical.last_digit_freq.most_common(10))

        breakdown = {}
        for name, pred in [
            ("statistical", stat_pred),
            ("rf+gb+xgb", tree_pred),
            ("lstm", lstm_pred),
            ("digit_position", dpos_pred),
        ]:
            if pred:
                simple = {}
                for k in PRIZE_KEYS:
                    if k in pred:
                        v = pred[k]
                        simple[k] = v if isinstance(v, str) else f"{len(v)} nums"
                if simple:
                    breakdown[name] = simple

        return {
            "hot_numbers": [str(n).zfill(4) for n in hot[:15]],
            "cold_numbers": [str(n).zfill(4) for n in cold[:15]],
            "last_digit_freq": {str(k): v for k, v in ldf.items()},
            "model_breakdown": breakdown,
            "cv_scores": {k: round(v, 4) for k, v in self.trees.cv_scores.items()},
        }
