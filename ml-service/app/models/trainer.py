import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
import json
import os
import hashlib
import logging
from datetime import date

logger = logging.getLogger(__name__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available, using statistical methods only")

PRIZE_KEYS = ["mc", "1st", "cons", "2nd", "3rd", "4th", "5th"]

# How many numbers each category has in a real Dear Lottery draw
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
    """Deterministic seed from date + slot so predictions are stable per day."""
    if date_str is None:
        date_str = str(date.today())
    raw = f"{date_str}:{time_slot}"
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


def extract_per_category(draws):
    """Extract number sequences per prize category across all draws."""
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


class StatisticalAnalyzer:
    """Per-category frequency and pattern analysis."""

    def __init__(self):
        self.category_freq = {}
        self.category_recent_freq = {}
        self.hot = {}
        self.cold = {}
        self.last_digit_freq = Counter()
        self.pair_freq = Counter()

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
            cold_candidates = overall_set - recent_set
            self.cold[key] = list(cold_candidates)[:15]

        all_first = [n for seq in category_sequences.get("1st", []) for n in seq]
        self.last_digit_freq = Counter(n % 10 for n in all_first)

        for seq in category_sequences.get("1st", []):
            for n in seq:
                d1 = (n // 10) % 10
                d2 = n % 10
                self.pair_freq[(d1, d2)] += 1

    def predict_numbers(self, key, count, rng):
        """Predict `count` unique numbers for a category using weighted sampling."""
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
        while len(results) < count and attempts < count * 20:
            chosen = rng.choice(numbers, p=weights)
            perturbation = rng.randint(-5, 6)
            result = (chosen + perturbation) % 10000
            results.add(result)
            attempts += 1

        # Fill remaining if needed
        while len(results) < count:
            results.add(rng.randint(0, 10000))

        return sorted([str(n).zfill(4) for n in results])

    def predict(self, rng):
        result = {}
        for key in PRIZE_KEYS:
            count = PRIZE_COUNTS[key]
            nums = self.predict_numbers(key, count, rng)
            result[key] = nums[0] if count == 1 else nums
        return result


class RFPredictor:
    """Random Forest per-category predictor."""

    def __init__(self):
        self.models = {}
        self.is_trained = False

    def fit(self, category_sequences):
        window = 5

        for key in PRIZE_KEYS:
            seqs = category_sequences.get(key, [])
            flat = [s[0] for s in seqs if s]

            if len(flat) < window + 5:
                continue

            X, y = [], []
            for i in range(window, len(flat)):
                features = flat[i - window : i]
                last_digits = [f % 10 for f in features]
                diffs = [features[j] - features[j - 1] for j in range(1, len(features))]
                X.append(features + last_digits + diffs)
                y.append(flat[i])

            model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
            model.fit(np.array(X), np.array(y))
            self.models[key] = {"model": model, "last_window": flat[-window:]}

        self.is_trained = bool(self.models)
        if self.is_trained:
            logger.info(f"RF trained for {list(self.models.keys())}")

    def predict_base(self, key):
        """Get a single base prediction for a category."""
        if key not in self.models:
            return None
        m = self.models[key]
        window = m["last_window"]
        last_digits = [f % 10 for f in window]
        diffs = [window[j] - window[j - 1] for j in range(1, len(window))]
        X = np.array(window + last_digits + diffs).reshape(1, -1)
        pred = m["model"].predict(X)[0]
        return int(pred) % 10000

    def predict_numbers(self, key, count, rng):
        """Predict `count` numbers by perturbing the base RF prediction."""
        base = self.predict_base(key)
        if base is None:
            return None

        results = {base}
        while len(results) < count:
            perturb = rng.randint(-50, 51)
            results.add((base + perturb) % 10000)

        return sorted([str(n).zfill(4) for n in results])

    def predict(self, rng):
        result = {}
        for key in PRIZE_KEYS:
            count = PRIZE_COUNTS[key]
            nums = self.predict_numbers(key, count, rng)
            if nums is not None:
                result[key] = nums[0] if count == 1 else nums
        return result


class LSTMPredictor:
    """LSTM per-category predictor."""

    def __init__(self):
        self.models = {}
        self.is_trained = False
        self.max_val = 10000

    def fit(self, category_sequences):
        if not TF_AVAILABLE:
            logger.warning("TF not available for LSTM")
            return

        window = 10

        for key in ["1st", "2nd", "3rd"]:
            seqs = category_sequences.get(key, [])
            flat = [s[0] for s in seqs if s]

            if len(flat) < window + 10:
                continue

            data = np.array(flat, dtype=np.float32) / self.max_val
            X, y = [], []
            for i in range(window, len(data)):
                X.append(data[i - window : i])
                y.append(data[i])

            X = np.array(X).reshape(-1, window, 1)
            y = np.array(y)

            model = keras.Sequential(
                [
                    keras.layers.LSTM(64, input_shape=(window, 1), return_sequences=True),
                    keras.layers.Dropout(0.2),
                    keras.layers.LSTM(32),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(16, activation="relu"),
                    keras.layers.Dense(1),
                ]
            )
            model.compile(optimizer="adam", loss="mse")
            model.fit(X, y, epochs=50, batch_size=16, verbose=0, validation_split=0.1)

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
            perturb = rng.randint(-50, 51)
            results.add((base + perturb) % 10000)

        return sorted([str(n).zfill(4) for n in results])

    def predict(self, rng):
        result = {}
        for key in PRIZE_KEYS:
            count = PRIZE_KEYS  # not used, just for top prizes
            nums = self.predict_numbers(key, PRIZE_COUNTS[key], rng)
            if nums is not None:
                result[key] = nums[0] if PRIZE_COUNTS[key] == 1 else nums
        return result


class EnsembleTrainer:
    """Manages all models and produces ensemble predictions with analysis data."""

    def __init__(self):
        self.statistical = StatisticalAnalyzer()
        self.rf = RFPredictor()
        self.lstm = LSTMPredictor()
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
        self.rf.fit(self.category_sequences)
        self.lstm.fit(self.category_sequences)

        self.is_trained = True
        return True

    def predict(self, time_slot="1pm", date_str=None):
        """Generate ensemble predictions. Same date+slot always gives same result."""
        seed = make_seed(time_slot, date_str)
        rng = np.random.RandomState(seed)

        stat_pred = self.statistical.predict(rng)
        rf_pred = self.rf.predict(rng) if self.rf.is_trained else {}
        lstm_pred = self.lstm.predict(rng) if self.lstm.is_trained else {}

        final = {}
        confidence_parts = []

        for key in PRIZE_KEYS:
            count = PRIZE_COUNTS[key]

            if count == 1:
                # Single number: weighted average of model base predictions
                candidates = []
                if key in stat_pred:
                    candidates.append(int(stat_pred[key]))
                if key in rf_pred:
                    candidates.append(int(rf_pred[key]))
                if key in lstm_pred:
                    candidates.append(int(lstm_pred[key]))

                if candidates:
                    if len(candidates) == 3:
                        w = [0.25, 0.4, 0.35]
                    elif len(candidates) == 2:
                        w = [0.35, 0.65]
                    else:
                        w = [1.0]
                    avg = int(np.average(candidates, weights=w)) % 10000
                    final[key] = str(avg).zfill(4)

                    model_count_bonus = len(candidates) * 0.15
                    if len(candidates) > 1:
                        spread = np.std(candidates) / 10000
                        agreement = max(0.1, 1 - spread * 2)
                    else:
                        agreement = 0.4
                    conf = min(0.95, agreement * 0.6 + model_count_bonus + 0.15)
                    confidence_parts.append(conf)
                else:
                    final[key] = str(rng.randint(0, 10000)).zfill(4)
                    confidence_parts.append(0.1)
            else:
                # Multiple numbers: merge from all models, pick top `count`
                all_nums = set()

                if key in stat_pred and isinstance(stat_pred[key], list):
                    all_nums.update(stat_pred[key])
                if key in rf_pred and isinstance(rf_pred[key], list):
                    all_nums.update(rf_pred[key])
                if key in lstm_pred and isinstance(lstm_pred[key], list):
                    all_nums.update(lstm_pred[key])

                # Fill if not enough
                freq = self.statistical.category_recent_freq.get(key, Counter())
                if freq:
                    numbers = list(freq.keys())
                    weights = np.array(list(freq.values()), dtype=np.float64)
                    weights = weights / weights.sum()
                    while len(all_nums) < count:
                        chosen = rng.choice(numbers, p=weights)
                        perturb = rng.randint(-10, 11)
                        all_nums.add(str((chosen + perturb) % 10000).zfill(4))

                while len(all_nums) < count:
                    all_nums.add(str(rng.randint(0, 10000)).zfill(4))

                final[key] = sorted(list(all_nums))[:count]

                # Confidence for multi-number categories
                sources = sum([
                    1 for src in [stat_pred, rf_pred, lstm_pred]
                    if key in src and isinstance(src[key], list) and len(src[key]) > 0
                ])
                conf = min(0.90, 0.3 + sources * 0.18)
                confidence_parts.append(conf)

        # Consolation = same last 4 digits as 1st prize
        if "1st" in final:
            final["cons"] = final["1st"]

        overall_confidence = float(np.mean(confidence_parts)) if confidence_parts else 0.1

        models_used = ["statistical"]
        if self.rf.is_trained:
            models_used.append("random_forest")
        if self.lstm.is_trained:
            models_used.append("lstm")

        analysis = self._build_analysis(stat_pred, rf_pred, lstm_pred)

        return {
            "predicted_numbers": final,
            "confidence": round(overall_confidence, 4),
            "model_used": "+".join(models_used),
            "analysis": analysis,
        }

    def _build_analysis(self, stat_pred, rf_pred, lstm_pred):
        hot = self.statistical.hot.get("1st", [])
        cold = self.statistical.cold.get("1st", [])

        ldf = dict(self.statistical.last_digit_freq.most_common(10))

        # Per-model breakdown: show only the single-number predictions for readability
        breakdown = {}
        for name, pred in [("statistical", stat_pred), ("random_forest", rf_pred), ("lstm", lstm_pred)]:
            if pred:
                simple = {}
                for k in PRIZE_KEYS:
                    if k in pred:
                        v = pred[k]
                        simple[k] = v if isinstance(v, str) else f"{len(v)} nums"
                breakdown[name] = simple

        return {
            "hot_numbers": [str(n).zfill(4) for n in hot[:15]],
            "cold_numbers": [str(n).zfill(4) for n in cold[:15]],
            "last_digit_freq": {str(k): v for k, v in ldf.items()},
            "model_breakdown": breakdown,
        }
