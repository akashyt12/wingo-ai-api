import json, numpy as np, pandas as pd, os, pickle, time, threading, queue, requests, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from collections import defaultdict, Counter
from flask import Flask, request, jsonify, Response

GAME_CODE = "WinGo_1M"
LOOKBACK = 10
MODEL_DIR = "models_wingo"
PAGE_SIZE = 50
FETCH_INTERVAL = 1
API_BASE = "https://draw.ar-lottery01.com/WinGo"
NUM_FEATURES = 30
BIG_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "big_dataset.json")

log_queues = []
def log(msg, level="info"):
    entry = json.dumps({"msg": str(msg), "level": level, "time": time.strftime("%H:%M:%S")})
    print(f"[{level.upper()}] {msg}")
    for q in list(log_queues):
        try: q.put_nowait(entry)
        except: pass

def fetch_history(game_code, page_size=50, max_pages=10):
    all_data = []
    for page in range(1, max_pages + 1):
        url = f"{API_BASE}/{game_code}/GetHistoryIssuePage.json?pageSize={page_size}&pageNo={page}"
        try:
            log(f"Fetching page {page}: {url}", "info")
            r = requests.get(url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate"
            })
            log(f"Response status: {r.status_code}", "info")
            if r.status_code != 200:
                log(f"HTTP {r.status_code}", "error")
                break
            j = r.json()
            items = j.get("data", {}).get("list", [])
            if not items:
                log(f"No items in page {page}", "info")
                break
            all_data.extend(items)
            log(f"Got {len(items)} items from page {page}", "success")
            time.sleep(0.2)
        except Exception as e:
            log(f"Fetch error page {page}: {e}", "error")
            break
    if not all_data:
        log("No data fetched from API", "error")
        return None
    df = pd.DataFrame(all_data)
    df['number'] = df['number'].astype(int)
    df['issueNumber'] = df['issueNumber'].astype(str)
    df = df.drop_duplicates(subset='issueNumber').sort_values('issueNumber').reset_index(drop=True)
    log(f"Total fetched: {len(df)} records", "success")
    return df[['issueNumber', 'number']]

def merge_csv(csv_path, new_df):
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path, dtype={'issueNumber': str})
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset='issueNumber').sort_values('issueNumber').reset_index(drop=True)
    combined.to_csv(csv_path, index=False)
    return combined

def load_big_dataset():
    if not os.path.exists(BIG_DATASET):
        return None
    try:
        log(f"Loading big_dataset.json...", "info")
        with open(BIG_DATASET, 'r') as f:
            data = json.load(f)
        records = data.get('records', [])
        if not records:
            return None
        df = pd.DataFrame(records)
        df['number'] = df['number'].astype(int)
        df['issueNumber'] = df['issueNumber'].astype(str)
        df = df[['issueNumber', 'number']].drop_duplicates(subset='issueNumber').sort_values('issueNumber').reset_index(drop=True)
        log(f"Big dataset: {len(df)} records loaded", "success")
        return df
    except Exception as e:
        log(f"Big dataset load error: {e}", "error")
        return None

def ensure_csv_ready():
    csv = "wingo_history.csv"
    if os.path.exists(csv):
        try:
            df = pd.read_csv(csv, dtype={'issueNumber': str})
            if len(df) > 50:
                log(f"CSV ready: {len(df)} records", "info")
                return csv
        except: pass
    log("Fetching from API...", "info")
    df = fetch_history(GAME_CODE, 50, 10)
    if df is not None and len(df) > 0:
        df.to_csv(csv, index=False)
        log(f"CSV from API: {len(df)} records", "success")
        return csv
    log("API failed - generating fallback data", "info")
    import random
    records = []
    base_issue = int(time.time()) % 10000000000
    for i in range(500):
        records.append({"issueNumber": str(base_issue + 500 - i), "number": random.randint(0, 9)})
    df = pd.DataFrame(records)
    df.to_csv(csv, index=False)
    log(f"Fallback CSV: {len(df)} records", "success")
    return csv

def extract_features(numbers):
    n = np.array(numbers, dtype=float)
    f = []
    f.extend(n[-LOOKBACK:].tolist())
    f.extend((n[-LOOKBACK:] / 9.0).tolist())
    for i in range(1, LOOKBACK):
        f.append(n[-i] - n[-i-1] if -i-1 >= -len(n) else 0)
    for d in range(10):
        f.append(float(np.sum(n[-LOOKBACK:] == d)))
    f.append(np.mean(n[-LOOKBACK:]))
    f.append(np.std(n[-LOOKBACK:]))
    f.append(np.min(n[-LOOKBACK:]))
    f.append(np.max(n[-LOOKBACK:]))
    bigs = [1 if x >= 5 else 0 for x in n[-LOOKBACK:]]
    f.append(sum(bigs))
    f.append(LOOKBACK - sum(bigs))
    streak = 1
    for i in range(len(n)-2, max(len(n)-LOOKBACK-1, -1), -1):
        if n[i] == n[i+1]: streak += 1
        else: break
    f.append(streak)
    last3 = n[-3:] if len(n) >= 3 else n
    f.append(np.mean(last3))
    f.append(np.std(last3))
    for d in range(10):
        f.append(1.0 if n[-1] == d else 0.0)
    while len(f) < NUM_FEATURES:
        f.append(0.0)
    return np.array(f[:NUM_FEATURES], dtype=np.float32)

def make_feature_sequences(numbers, lookback):
    X, y = [], []
    for i in range(lookback, len(numbers)):
        seq = numbers[max(0, i-lookback):i]
        if len(seq) >= lookback:
            feat = extract_features(seq)
            X.append(feat)
            y.append(1 if numbers[i] >= 5 else 0)
    return np.array(X), np.array(y)

class MetaBandit:
    def __init__(self, n=4, c=2.0):
        self.n, self.c = n, c
        self.counts = np.zeros(n)
        self.values = np.zeros(n)
    def select(self):
        t = np.sum(self.counts)
        if t < self.n: return int(t) % self.n
        ucb = self.values + self.c * np.sqrt(np.log(t + 1) / (self.counts + 1))
        return int(np.argmax(ucb))
    def update(self, i, reward):
        self.counts[i] += 1
        self.values[i] += (reward - self.values[i]) / self.counts[i]
    def best(self): return int(np.argmax(self.values))
    def stats(self): return {"counts": self.counts.tolist(), "values": [round(float(v),4) for v in self.values]}

class WinGoAI:
    def __init__(self):
        self.lb = LOOKBACK
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.rf_m = None
        self.gb_m = None
        self.mlp_m = None
        self.scaler = None
        self.is_training = False
        self.train_pct = 0
        self.is_live = False
        self.last_pred = None
        self.total_records = 0
        self.stats = {"wins":0, "losses":0, "total":0, "accuracy":0}
        self.ready = False
        self.next_period = "--"
        self.all_numbers = []
        self.pattern_map = defaultdict(lambda: [0, 0])
        self.transitions = np.zeros((10, 10))
        self.h2_transitions = defaultdict(lambda: np.zeros(10))
        self.streak_map = defaultdict(lambda: [0, 0])
        self.sum_map = defaultdict(lambda: [0, 0])
        self.bootstrap_map = defaultdict(lambda: [0, 0])
        self.FREQ_WINDOWS = [5, 8, 12, 20]
        self.live_interval = 30

    def _build_pattern_engine(self, nums):
        t0 = time.time()
        self.transitions = np.zeros((10, 10))
        self.h2_transitions = defaultdict(lambda: np.zeros(10))
        self.pattern_map = defaultdict(lambda: [0, 0])
        self.streak_map = defaultdict(lambda: [0, 0])
        self.sum_map = defaultdict(lambda: [0, 0])
        self.bootstrap_map = defaultdict(lambda: [0, 0])
        PATTERN_TRAIN = 50000
        pn = nums[-PATTERN_TRAIN:] if len(nums) > PATTERN_TRAIN else nums
        log(f"Pattern Engine: using last {len(pn)} records", "info")
        for i in range(1, len(pn)):
            self.transitions[pn[i-1]][pn[i]] += 1
        for i in range(2, len(pn)):
            key = (pn[i-2], pn[i-1])
            self.h2_transitions[key][pn[i]] += 1
        for size in [3, 4, 5]:
            for i in range(size, len(pn)):
                key = tuple(pn[i-size:i])
                label = 1 if pn[i] >= 5 else 0
                self.pattern_map[key][label] += 1
        streak_len = 1
        for i in range(1, len(pn)):
            if pn[i] == pn[i-1]:
                streak_len += 1
            else:
                key = (pn[i-1], streak_len)
                self.streak_map[key][0 if pn[i] < 5 else 1] += 1
                streak_len = 1
        for i in range(3, len(pn)):
            s = sum(pn[i-3:i])
            self.sum_map[s][0 if pn[i] < 5 else 1] += 1
        for size in [3, 4, 5]:
            for i in range(size, len(pn)):
                key = tuple(pn[i-size:i])
                label = 1 if pn[i] >= 5 else 0
                bkey = f"bs{size}_{key}"
                self.bootstrap_map[bkey][label] += 1
        log(f"Pattern Engine: {time.time()-t0:.1f}s | {len(self.pattern_map)} patterns", "success")

    def _predict_markov1(self, history):
        if len(history) < 2: return None, 0
        trans = self.transitions[history[-1]]
        total = sum(trans)
        if total < 5: return None, 0
        big_prob = sum(trans[5:]) / total
        return 1 if big_prob > 0.5 else 0, abs(big_prob - 0.5) * 2

    def _predict_markov2(self, history):
        if len(history) < 3: return None, 0
        key = (history[-2], history[-1])
        if key not in self.h2_transitions: return None, 0
        trans = self.h2_transitions[key]
        total = sum(trans)
        if total < 3: return None, 0
        big_prob = sum(trans[5:]) / total
        return 1 if big_prob > 0.5 else 0, abs(big_prob - 0.5) * 2.5

    def _predict_tuple(self, history, size=3):
        if len(history) < size + 1: return None, 0
        key = tuple(history[-size:])
        if key not in self.pattern_map: return None, 0
        counts = self.pattern_map[key]
        total = sum(counts)
        if total < 2: return None, 0
        big_prob = counts[1] / total
        return 1 if counts[1] > counts[0] else 0, abs(big_prob - 0.5) * (3 - size/2)

    def _predict_streak(self, history):
        if len(history) < 3: return None, 0
        streak = 1
        for i in range(len(history)-2, max(len(history)-10, -1), -1):
            if history[i] == history[i+1]: streak += 1
            else: break
        key = (history[-1], streak)
        if key not in self.streak_map: return None, 0
        counts = self.streak_map[key]
        total = sum(counts)
        if total < 3: return None, 0
        big_prob = counts[1] / total
        return 1 if counts[1] > counts[0] else 0, abs(big_prob - 0.5) * 2

    def _predict_sum(self, history):
        if len(history) < 4: return None, 0
        s = sum(history[-3:])
        if s not in self.sum_map: return None, 0
        counts = self.sum_map[s]
        total = sum(counts)
        if total < 5: return None, 0
        big_prob = counts[1] / total
        return 1 if counts[1] > counts[0] else 0, abs(big_prob - 0.5) * 1.5

    def _predict_bootstrap(self, history):
        votes = []
        for size in [3, 4, 5]:
            if len(history) >= size + 1:
                key = tuple(history[-size:])
                bkey = f"bs{size}_{key}"
                if bkey in self.bootstrap_map:
                    counts = self.bootstrap_map[bkey]
                    total = sum(counts)
                    if total >= 2:
                        p = counts[1] / total
                        votes.append((1 if p > 0.5 else 0, abs(p - 0.5)))
        if not votes: return None, 0
        total_conf = sum(c for _, c in votes)
        if total_conf == 0: return None, 0
        weighted = sum(p * c for p, c in votes) / total_conf
        return 1 if weighted > 0.5 else 0, weighted

    def _pattern_ensemble(self, history):
        predictions = []
        for name, func in [
            ("m1", self._predict_markov1), ("m2", self._predict_markov2),
            ("t3", lambda h: self._predict_tuple(h, 3)),
            ("t4", lambda h: self._predict_tuple(h, 4)),
            ("t5", lambda h: self._predict_tuple(h, 5)),
            ("str", self._predict_streak), ("sum", self._predict_sum),
            ("pos", self._predict_markov1), ("bs", self._predict_bootstrap),
        ]:
            p, c = func(history)
            if p is not None: predictions.append((p, c))
        if not predictions: return None, 0, []
        total_w = sum(c for _, c in predictions)
        weighted = sum(p * c for p, c in predictions) / total_w
        pred = 1 if weighted > 0.5 else 0
        agree = sum(1 for p, _ in predictions if p == pred)
        return pred, agree / len(predictions), [p for p, _ in predictions]

    def _freq_simple(self, history, window=8):
        window = min(window, len(history))
        recent = history[-window:]
        big_c = sum(1 for x in recent if x >= 5)
        return 1 if big_c > window / 2 else 0

    def train_all(self, csv):
        self.is_training = True
        self.train_pct = 0
        log("=" * 50, "info")
        log("  TRAINING - POWERFUL ENSEMBLE v2.0", "info")
        log("=" * 50, "info")

        if not os.path.exists(csv):
            log(f"CSV not found: {csv}", "error")
            self.is_training = False
            return

        df = pd.read_csv(csv)
        all_nums = df['number'].values.astype(int).tolist()
        self.all_numbers = all_nums
        self.total_records = len(all_nums)
        log(f"Dataset: {len(all_nums)} records", "info")
        self.train_pct = 10

        self._build_pattern_engine(all_nums)
        self.train_pct = 30

        ML_TRAIN = 5000
        train_nums = all_nums[-ML_TRAIN:] if len(all_nums) > ML_TRAIN else all_nums
        log(f"ML: last {len(train_nums)} records", "info")

        X, y = make_feature_sequences(train_nums, self.lb)
        self.train_pct = 50

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.rf_m = RandomForestClassifier(n_estimators=150, max_depth=10, n_jobs=-1, random_state=42)
        self.rf_m.fit(X_scaled, y)
        log(f"  RF trained: {len(X)} samples", "info")

        self.gb_m = GradientBoostingClassifier(n_estimators=80, max_depth=5, random_state=42)
        self.gb_m.fit(X_scaled, y)
        log(f"  GB trained: {len(X)} samples", "info")

        self.mlp_m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
        self.mlp_m.fit(X_scaled, y)
        log(f"  MLP trained: {len(X)} samples", "info")

        with open(f'{MODEL_DIR}/rf_opt.pkl', 'wb') as f: pickle.dump(self.rf_m, f)
        with open(f'{MODEL_DIR}/gb_opt.pkl', 'wb') as f: pickle.dump(self.gb_m, f)
        with open(f'{MODEL_DIR}/mlp_opt.pkl', 'wb') as f: pickle.dump(self.mlp_m, f)
        with open(f'{MODEL_DIR}/scaler_opt.pkl', 'wb') as f: pickle.dump(self.scaler, f)
        with open(f'{MODEL_DIR}/patterns.pkl', 'wb') as f: pickle.dump(dict(self.pattern_map), f)
        with open(f'{MODEL_DIR}/transitions.pkl', 'wb') as f: pickle.dump(self.transitions, f)
        with open(f'{MODEL_DIR}/h2_trans.pkl', 'wb') as f: pickle.dump(dict(self.h2_transitions), f)
        with open(f'{MODEL_DIR}/streak_map.pkl', 'wb') as f: pickle.dump(dict(self.streak_map), f)
        with open(f'{MODEL_DIR}/sum_map.pkl', 'wb') as f: pickle.dump(dict(self.sum_map), f)
        with open(f'{MODEL_DIR}/bootstrap_map.pkl', 'wb') as f: pickle.dump(dict(self.bootstrap_map), f)
        self.train_pct = 80

        log("Testing on last 100...", "info")
        test_start = max(0, len(all_nums) - 100)
        test_history = list(all_nums[:test_start])
        wins = 0
        for i in range(test_start, len(all_nums)):
            actual = all_nums[i]
            pred = self._ensemble_predict(test_history)
            actual_label = 1 if actual >= 5 else 0
            if pred == actual_label:
                wins += 1
            test_history.append(actual)
        acc = (wins / (len(all_nums) - test_start)) * 100
        log(f"Test: {wins}/{len(all_nums) - test_start} = {acc:.1f}%", "success")
        self.train_pct = 100
        self.ready = True

        self.stats = {"wins": 0, "losses": 0, "total": 0, "accuracy": round(acc, 1)}
        log("=" * 50, "success")
        log(f"  TRAINING DONE! Accuracy: {acc:.1f}%", "success")
        log("=" * 50, "success")
        self.is_training = False

    def load(self):
        try:
            with open(f'{MODEL_DIR}/rf_opt.pkl', 'rb') as f: self.rf_m = pickle.load(f)
            with open(f'{MODEL_DIR}/scaler_opt.pkl', 'rb') as f: self.scaler = pickle.load(f)
            with open(f'{MODEL_DIR}/patterns.pkl', 'rb') as f: self.pattern_map = defaultdict(lambda: [0, 0], pickle.load(f))
            if os.path.exists(f'{MODEL_DIR}/gb_opt.pkl'):
                with open(f'{MODEL_DIR}/gb_opt.pkl', 'rb') as f: self.gb_m = pickle.load(f)
            if os.path.exists(f'{MODEL_DIR}/mlp_opt.pkl'):
                with open(f'{MODEL_DIR}/mlp_opt.pkl', 'rb') as f: self.mlp_m = pickle.load(f)
            if os.path.exists(f'{MODEL_DIR}/transitions.pkl'):
                with open(f'{MODEL_DIR}/transitions.pkl', 'rb') as f: self.transitions = pickle.load(f)
            if os.path.exists(f'{MODEL_DIR}/h2_trans.pkl'):
                with open(f'{MODEL_DIR}/h2_trans.pkl', 'rb') as f: self.h2_transitions = defaultdict(lambda: np.zeros(10), pickle.load(f))
            if os.path.exists(f'{MODEL_DIR}/streak_map.pkl'):
                with open(f'{MODEL_DIR}/streak_map.pkl', 'rb') as f: self.streak_map = defaultdict(lambda: [0, 0], pickle.load(f))
            if os.path.exists(f'{MODEL_DIR}/sum_map.pkl'):
                with open(f'{MODEL_DIR}/sum_map.pkl', 'rb') as f: self.sum_map = defaultdict(lambda: [0, 0], pickle.load(f))
            if os.path.exists(f'{MODEL_DIR}/bootstrap_map.pkl'):
                with open(f'{MODEL_DIR}/bootstrap_map.pkl', 'rb') as f: self.bootstrap_map = defaultdict(lambda: [0, 0], pickle.load(f))
            self.ready = True
            log("Powerful Ensemble v2.0 loaded (RF+GB+MLP+Markov+Patterns+Streak+Sum)", "success")
            return True
        except Exception as e:
            log(f"Load failed: {e}", "error")
            return False

    def predict(self, seq):
        if len(seq) != self.lb:
            raise ValueError(f"Need {self.lb} numbers")
        if not self.ready:
            raise ValueError("Not trained yet.")
        self.all_numbers = seq
        self.next_period = str(int(time.time()) % 10000).zfill(4)

        pe_pred, pe_conf, pe_votes = self._pattern_ensemble(seq)

        feat = extract_features(seq).reshape(1, -1)
        feat_s = self.scaler.transform(feat)
        rf_pred = int(self.rf_m.predict(feat_s)[0]) if self.rf_m else None
        gb_pred = int(self.gb_m.predict(feat_s)[0]) if self.gb_m else None
        mlp_pred = int(self.mlp_m.predict(feat_s)[0]) if self.mlp_m else None

        freq_preds = []
        for w in self.FREQ_WINDOWS:
            freq_preds.append(self._freq_simple(seq, w))

        vote_list = []
        if pe_pred is not None:
            vote_list.append(("Patterns", "big" if pe_pred == 1 else "small"))
        if rf_pred is not None:
            vote_list.append(("RF", "big" if rf_pred == 1 else "small"))
        if gb_pred is not None:
            vote_list.append(("GB", "big" if gb_pred == 1 else "small"))
        if mlp_pred is not None:
            vote_list.append(("MLP", "big" if mlp_pred == 1 else "small"))
        for idx, fp in enumerate(freq_preds):
            vote_list.append((f"Freq{self.FREQ_WINDOWS[idx]}", "big" if fp == 1 else "small"))

        big_count = sum(1 for sz, _ in vote_list if sz == "big")
        small_count = sum(1 for sz, _ in vote_list if sz == "small")
        total_votes = len(vote_list)
        agree_count = max(big_count, small_count)
        size = "big" if big_count > small_count else "small"

        should_skip = False
        confidence = round((agree_count / total_votes) * 100) if total_votes > 0 else 0

        r = {
            "size": size,
            "number": 7 if size == "big" else 2,
            "color": "green" if size == "big" else "red",
            "skip": False,
            "confidence": confidence,
            "agree_count": agree_count,
            "total_votes": total_votes,
            "models_voted": [
                {"name": name, "size": sz, "number": 7 if sz == "big" else 2}
                for name, sz in vote_list
            ],
            "features": NUM_FEATURES,
            "based_on": seq,
        }
        r["message"] = f"{size.upper()} ({agree_count}/{total_votes} agree)"
        log(f"PREDICT: {size.upper()} | {agree_count}/{total_votes} agree", "success")
        self.last_pred = r
        return r

    def _ensemble_predict(self, seq):
        pe_pred, _, _ = self._pattern_ensemble(seq)
        feat = extract_features(seq).reshape(1, -1)
        feat_s = self.scaler.transform(feat)
        rf_pred = int(self.rf_m.predict(feat_s)[0]) if self.rf_m else None
        gb_pred = int(self.gb_m.predict(feat_s)[0]) if self.gb_m else None
        mlp_pred = int(self.mlp_m.predict(feat_s)[0]) if self.mlp_m else None
        freq_preds = [self._freq_simple(seq, w) for w in self.FREQ_WINDOWS]
        votes = []
        if pe_pred is not None: votes.append(pe_pred)
        if rf_pred is not None: votes.append(rf_pred)
        if gb_pred is not None: votes.append(gb_pred)
        if mlp_pred is not None: votes.append(mlp_pred)
        votes.extend(freq_preds)
        return 1 if sum(votes) > len(votes) / 2 else 0

    def quick_check(self, seq, actual_number):
        pred = self._ensemble_predict(seq)
        actual = int(actual_number)
        predicted_size = "big" if pred == 1 else "small"
        actual_size = "big" if actual >= 5 else "small"
        is_win = predicted_size == actual_size
        return {
            "predicted_size": predicted_size, "actual_number": actual,
            "actual_size": actual_size, "is_win": is_win,
            "stats": self.stats
        }

    def update(self, seq, true_num):
        if not self.ready:
            return
        true_num = int(true_num)
        self.all_numbers = seq + [true_num]
        self.stats["total"] += 1
        if self.last_pred:
            pred_size = self.last_pred.get("size", "small")
            actual_size = "big" if true_num >= 5 else "small"
            if pred_size == actual_size:
                self.stats["wins"] += 1
            else:
                self.stats["losses"] += 1
            total = self.stats["total"]
            if total > 0:
                self.stats["accuracy"] = round((self.stats["wins"] / total) * 100, 1)
        key = tuple(seq[-3:]) if len(seq) >= 3 else None
        if key:
            self.pattern_map[key][0 if true_num < 5 else 1] += 1

    def status(self):
        return {
            "models_loaded": self.ready,
            "best_model": "Ensemble v2.0" if self.ready else "none",
            "is_training": self.is_training, "train_pct": self.train_pct,
            "is_live": self.is_live, "total_records": self.total_records,
            "stats": self.stats,
            "last_pred": self.last_pred, "features": NUM_FEATURES,
            "next_period": self.next_period,
            "pattern_count": len(self.pattern_map),
            "freq_windows": self.FREQ_WINDOWS,
            "live_interval": self.live_interval
        }

app = Flask(__name__)
ai = WinGoAI()

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>WinGo AI</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0a0a0f;--card:#12121a;--border:#1e1e2e;--accent:#6c5ce7;--green:#00b894;--red:#e17055;--text:#f0f0f5;--dim:#6c7a8a;--glow:rgba(108,92,231,.3)}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100dvh;max-width:480px;margin:0 auto;padding:12px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px;margin-bottom:10px}
.label{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:1px;font-weight:600}
.mono{font-family:'JetBrains Mono',monospace}

/* Status Bar */
.status{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:10px}
.stat{text-align:center;padding:8px 4px;background:var(--card);border:1px solid var(--border);border-radius:8px}
.stat .val{font-size:16px;font-weight:700;font-family:'JetBrains Mono',monospace}
.stat .lbl{font-size:8px;color:var(--dim);text-transform:uppercase;letter-spacing:.5px;margin-top:2px}

/* Big Prediction */
.pred{position:relative;text-align:center;padding:20px 0}
.pred .size{font-size:14px;font-weight:600;text-transform:uppercase;letter-spacing:2px;margin-bottom:4px}
.pred .number{font-size:100px;font-weight:800;font-family:'JetBrains Mono',monospace;line-height:1;transition:all .3s}
.pred .number.big{color:var(--green);text-shadow:0 0 50px rgba(0,184,148,.4)}
.pred .number.small{color:var(--red);text-shadow:0 0 50px rgba(225,112,85,.4)}
.pred .color{font-size:12px;font-weight:600;margin-top:4px}
.pred .color.green{color:var(--green)}.pred .color.red{color:var(--red)}
.pred .vote-tag{display:inline-block;padding:4px 10px;border-radius:6px;font-size:9px;font-weight:600;margin-top:8px;background:rgba(108,92,231,.15);border:1px solid rgba(108,92,231,.3);color:var(--accent)}

/* Votes */
.votes{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin:10px 0}
.vc{padding:8px 4px;text-align:center;background:var(--bg);border:1px solid var(--border);border-radius:8px}
.vc .vn{font-size:7px;color:var(--dim);text-transform:uppercase;letter-spacing:.5px;font-weight:600}
.vc .vs{font-size:11px;font-weight:700;margin-top:2px;font-family:'JetBrains Mono',monospace}
.vc .vs.big{color:var(--green)}.vc .vs.small{color:var(--red)}

/* History */
.hist{display:flex;gap:4px;overflow-x:auto;padding:4px 0;scrollbar-width:none}
.hist::-webkit-scrollbar{display:none}
.hi{min-width:28px;height:28px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;flex-shrink:0}
.hi.big{background:rgba(0,184,148,.15);color:var(--green);border:1px solid rgba(0,184,148,.2)}
.hi.small{background:rgba(225,112,85,.15);color:var(--red);border:1px solid rgba(225,112,85,.2)}

/* Win/Loss */
.wl{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin:8px 0}
.wl-item{padding:10px;text-align:center;border-radius:8px}
.wl-item.win{background:rgba(0,184,148,.08);border:1px solid rgba(0,184,148,.2)}
.wl-item.loss{background:rgba(225,112,85,.08);border:1px solid rgba(225,112,85,.2)}
.wl-item.total{background:rgba(108,92,231,.08);border:1px solid rgba(108,92,231,.2)}
.wl-item .wv{font-size:22px;font-weight:700;font-family:'JetBrains Mono',monospace}
.wl-item.win .wv{color:var(--green)}.wl-item.loss .wv{color:var(--red)}.wl-item.total .wv{color:var(--accent)}
.wl-item .wl{font-size:8px;color:var(--dim);text-transform:uppercase;letter-spacing:.5px;margin-top:2px;font-weight:600}

/* Last Result */
.last-result{display:flex;align-items:center;justify-content:center;gap:12px;padding:10px;background:var(--bg);border:1px solid var(--border);border-radius:8px;margin:8px 0}
.last-result .lr-num{font-size:40px;font-weight:700;font-family:'JetBrains Mono',monospace}
.last-result .lr-info{text-align:left}
.last-result .lr-info div{font-size:10px;color:var(--dim);margin:2px 0}

/* Log */
.logbox{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:8px;max-height:150px;overflow-y:auto;font-family:'JetBrains Mono',monospace;font-size:9px;line-height:1.7}
.logbox .l{opacity:0;animation:fadeIn .3s forwards}
@keyframes fadeIn{to{opacity:1}}
.l-time{color:var(--dim)}.l-info{color:var(--dim)}.l-success{color:var(--green)}.l-error{color:var(--red)}

/* Status Badge */
.badge{display:inline-flex;align-items:center;gap:4px;padding:3px 8px;border-radius:10px;font-size:8px;font-weight:600}
.badge.live{background:rgba(0,184,148,.12);color:var(--green);border:1px solid rgba(0,184,148,.25)}
.badge.off{background:rgba(108,92,231,.12);color:var(--accent);border:1px solid rgba(108,92,231,.25)}
.pulse{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(0,184,148,.4)}50%{box-shadow:0 0 0 6px rgba(0,184,148,0)}}

/* API Info */
.api-box{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:8px;font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--dim);word-break:break-all}
.api-box .method{color:var(--green);font-weight:600}
.api-box .path{color:var(--accent)}
</style>
</head>
<body>

<!-- Status Bar -->
<div class="status">
  <div class="stat"><div class="val mono" id="sR">0</div><div class="lbl">Records</div></div>
  <div class="stat"><div class="val mono" id="sW" style="color:var(--green)">0</div><div class="lbl">Wins</div></div>
  <div class="stat"><div class="val mono" id="sA" style="color:var(--accent)">0%</div><div class="lbl">Accuracy</div></div>
  <div class="stat"><div class="val mono" id="sL" style="color:var(--red)">0</div><div class="lbl">Losses</div></div>
</div>

<!-- Live Status -->
<div style="text-align:center;margin-bottom:10px">
  <span class="badge off" id="liveBadge"><span class="pulse" style="background:var(--accent)"></span>&nbsp; STOPPED</span>
</div>

<!-- Interval Buttons -->
<div style="display:flex;gap:6px;justify-content:center;margin-bottom:10px">
  <button onclick="startLive(30)" id="btn30s" style="padding:6px 14px;border-radius:8px;border:1px solid var(--border);background:var(--card);color:var(--text);font-size:10px;font-weight:600;cursor:pointer;font-family:'Inter',sans-serif">30s</button>
  <button onclick="startLive(60)" id="btn60s" style="padding:6px 14px;border-radius:8px;border:1px solid var(--border);background:var(--card);color:var(--text);font-size:10px;font-weight:600;cursor:pointer;font-family:'Inter',sans-serif">1min</button>
  <button onclick="stopLive()" id="btnStop" style="padding:6px 14px;border-radius:8px;border:1px solid rgba(225,112,85,.3);background:rgba(225,112,85,.1);color:var(--red);font-size:10px;font-weight:600;cursor:pointer;font-family:'Inter',sans-serif">Stop</button>
</div>

<!-- Main Prediction -->
<div class="card">
  <div class="pred">
    <div class="size" id="predSize">--</div>
    <div class="number" id="predNum">?</div>
    <div class="color" id="predColor">--</div>
    <div class="vote-tag" id="predVote">Waiting for data...</div>
  </div>
</div>

<!-- Last Result -->
<div class="card">
  <div class="label" style="margin-bottom:6px"><i class="fas fa-circle-check"></i> Last Result</div>
  <div class="last-result">
    <div class="lr-num" id="lrNum">?</div>
    <div class="lr-info">
      <div>Size: <strong id="lrSize">--</strong></div>
      <div>Color: <strong id="lrColor">--</strong></div>
      <div>Issue: <span id="lrIssue" class="mono" style="color:var(--dim)">--</span></div>
    </div>
  </div>
</div>

<!-- Vote Breakdown -->
<div class="card">
  <div class="label" style="margin-bottom:6px"><i class="fas fa-users"></i> Vote Breakdown</div>
  <div class="votes">
    <div class="vc"><div class="vn">Patterns</div><div class="vs" id="vPatterns">--</div></div>
    <div class="vc"><div class="vn">RF</div><div class="vs" id="vRF">--</div></div>
    <div class="vc"><div class="vn">GB</div><div class="vs" id="vGB">--</div></div>
    <div class="vc"><div class="vn">MLP</div><div class="vs" id="vMLP">--</div></div>
  </div>
  <div class="votes" style="margin-top:4px">
    <div class="vc"><div class="vn">Freq5</div><div class="vs" id="vFreq5">--</div></div>
    <div class="vc"><div class="vn">Freq8</div><div class="vs" id="vFreq8">--</div></div>
    <div class="vc"><div class="vn">Freq12</div><div class="vs" id="vFreq12">--</div></div>
    <div class="vc"><div class="vn">Freq20</div><div class="vs" id="vFreq20">--</div></div>
  </div>
</div>

<!-- History -->
<div class="card">
  <div class="label" style="margin-bottom:6px"><i class="fas fa-clock-rotate-left"></i> History</div>
  <div class="hist" id="hist"></div>
</div>

<!-- Win/Loss -->
<div class="card">
  <div class="label" style="margin-bottom:6px"><i class="fas fa-chart-bar"></i> Win/Loss</div>
  <div class="wl">
    <div class="wl-item win"><div class="wv" id="wW">0</div><div class="wl">Wins</div></div>
    <div class="wl-item loss"><div class="wv" id="wL">0</div><div class="wl">Losses</div></div>
    <div class="wl-item total"><div class="wv" id="wT">0</div><div class="wl">Total</div></div>
  </div>
</div>

<!-- Log -->
<div class="card">
  <div class="label" style="margin-bottom:6px"><i class="fas fa-scroll"></i> Log</div>
  <div class="logbox" id="logbox"></div>
</div>

<!-- API Info -->
<div class="card">
  <div class="label" style="margin-bottom:6px"><i class="fas fa-code"></i> API Endpoints</div>
  <div class="api-box">
    <div><span class="method">GET</span> <span class="path">/status</span> - Current status</div>
    <div><span class="method">GET</span> <span class="path">/history</span> - Last results</div>
    <div><span class="method">POST</span> <span class="path">/predict</span> - Predict from numbers</div>
    <div><span class="method">POST</span> <span class="path">/update</span> - Update with actual</div>
    <div><span class="method">POST</span> <span class="path">/train</span> - Start training</div>
    <div><span class="method">POST</span> <span class="path">/live</span> - Start live mode</div>
  </div>
</div>

<script>
let sse,logs=[];
const LB=10;
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function sn(n){return n>=5?'Big':'Small'}
function cn(n){return n%2===1?'Green':'Red'}
function sc(n){return n>=5?'big':'small'}
function cc(n){return n%2===1?'green':'red'}
function connSSE(){if(sse)sse.close();sse=new EventSource('/stream_logs');sse.onmessage=e=>{try{const d=JSON.parse(e.data);aLog(d.msg,d.level)}catch(x){}};sse.onerror=()=>setTimeout(connSSE,3000)}
function aLog(m,l='info'){const t=new Date().toLocaleTimeString();logs.push({t,m,l});if(logs.length>100)logs.shift();rLog()}
function rLog(){const el=document.getElementById('logbox');el.innerHTML=logs.slice(-40).map(l=>`<div class="l"><span class="l-time">[${l.t}]</span> <span class="l-${l.l}">${esc(l.m)}</span></div>`).join('');el.scrollTop=el.scrollHeight}

function showP(d){
  if(!d)return;
  const sz=sc(d.number),col=cc(d.number);
  document.getElementById('predSize').textContent=sn(d.number).toUpperCase();
  document.getElementById('predSize').style.color=col==='green'?'var(--green)':'var(--red)';
  document.getElementById('predNum').textContent=d.number;
  document.getElementById('predNum').className='number '+sz;
  document.getElementById('predColor').textContent=cn(d.number);
  document.getElementById('predColor').className='color '+col;
  document.getElementById('predVote').textContent=d.message||`${d.agree_count||0}/${d.total_votes||0} agree`;
  document.getElementById('predVote').style.background='rgba(0,184,148,.15)';
  document.getElementById('predVote').style.borderColor='rgba(0,184,148,.3)';
  document.getElementById('predVote').style.color='var(--green)';
  if(d.models_voted){
    const map={'Patterns':'vPatterns','RF':'vRF','GB':'vGB','MLP':'vMLP','Freq5':'vFreq5','Freq8':'vFreq8','Freq12':'vFreq12','Freq20':'vFreq20'};
    d.models_voted.forEach(m=>{const el=document.getElementById(map[m.name]);if(el){el.textContent=m.size==='big'?'BIG':'SMALL';el.className='vs '+m.size}});
  }
}

function rH(h){document.getElementById('hist').innerHTML=h.slice(-30).reverse().map(x=>`<div class="hi ${sc(x.number)}">${x.number}</div>`).join('')}
function uS(s){
  document.getElementById('sR').textContent=s.total_records||0;
  document.getElementById('sW').textContent=s.stats.wins||0;
  document.getElementById('sA').textContent=(s.stats.accuracy||0)+'%';
  document.getElementById('sL').textContent=s.stats.losses||0;
  document.getElementById('wW').textContent=s.stats.wins||0;
  document.getElementById('wL').textContent=s.stats.losses||0;
  document.getElementById('wT').textContent=s.stats.total||0;
  const badge=document.getElementById('liveBadge');
  const interval=s.live_interval||30;
  if(s.is_live){badge.className='badge live';badge.innerHTML=`<span class="pulse"></span>&nbsp; LIVE ${interval}s`}
  else{badge.className='badge off';badge.innerHTML='<span class="pulse" style="background:var(--accent)"></span>&nbsp; STOPPED'}
  const btn30=document.getElementById('btn30s');
  const btn60=document.getElementById('btn60s');
  if(s.is_live){
    btn30.style.background=interval===30?'rgba(0,184,148,.2)':'var(--card)';
    btn30.style.borderColor=interval===30?'var(--green)':'var(--border)';
    btn30.style.color=interval===30?'var(--green)':'var(--text)';
    btn60.style.background=interval===60?'rgba(0,184,148,.2)':'var(--card)';
    btn60.style.borderColor=interval===60?'var(--green)':'var(--border)';
    btn60.style.color=interval===60?'var(--green)':'var(--text)';
  }else{
    btn30.style.background='var(--card)';btn30.style.borderColor='var(--border)';btn30.style.color='var(--text)';
    btn60.style.background='var(--card)';btn60.style.borderColor='var(--border)';btn60.style.color='var(--text)';
  }
  if(s.last_pred)showP(s.last_pred);
}
function uLR(d){
  if(!d)return;
  document.getElementById('lrNum').textContent=d.number;
  document.getElementById('lrNum').style.color=d.number>=5?'var(--green)':'var(--red)';
  document.getElementById('lrSize').textContent=sn(d.number);
  document.getElementById('lrSize').style.color=d.number>=5?'var(--green)':'var(--red)';
  document.getElementById('lrColor').textContent=cn(d.number);
  document.getElementById('lrColor').style.color=d.number%2===1?'var(--green)':'var(--red)';
  document.getElementById('lrIssue').textContent=d.issue||'--';
}
async function api(u,m='GET',b=null){const o={method:m,headers:{'Content-Type':'application/json'}};if(b)o.body=JSON.stringify(b);return(await fetch(u,o)).json()}
async function poll(){try{const s=await api('/status');uS(s);const lr=await api('/last_result');if(lr)uLR(lr)}catch(e){}}

async function startLive(interval){
  try{
    aLog(`Starting live (${interval}s)...`,'info');
    await api('/live','POST',{interval:interval});
    aLog(`Live ON! ${interval}s polling`,'success');
  }catch(e){aLog('Error: '+e.message,'error')}
}
async function stopLive(){
  try{
    await api('/live_stop','POST');
    aLog('Live STOPPED','info');
  }catch(e){aLog('Error: '+e.message,'error')}
}

async function autoStart(){
  try{
    aLog('Auto-starting...','info');
    let s=await api('/status');
    if(!s.models_loaded){
      aLog('Training models...','info');
      await api('/train','POST');
      let tries=0;
      while(tries<30){await new Promise(r=>setTimeout(r,1000));s=await api('/status');if(!s.is_training){aLog('Training done!','success');break}tries++}
    }
    aLog('Starting live (30s)...','info');
    await api('/live','POST',{interval:30});
    aLog('Auto-started!','success');
  }catch(e){aLog('Auto-start error: '+e.message,'error')}
}

connSSE();poll();setInterval(poll,3000);
autoStart();
</script>
</body>
</html>'''

@app.route('/')
def index(): return Response(HTML, mimetype='text/html')

@app.route('/stream_logs')
def stream_logs():
    q = queue.Queue(); log_queues.append(q)
    def gen():
        try:
            yield f"data: {json.dumps({'msg':'Connected','level':'success','time':time.strftime('%H:%M:%S')})}\n\n"
            while True:
                try: data = q.get(timeout=30); yield f"data: {data}\n\n"
                except queue.Empty: yield ": ka\n\n"
        except GeneratorExit:
            if q in log_queues: log_queues.remove(q)
    return Response(gen(), mimetype='text/event-stream', headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

@app.route('/status')
def status(): return jsonify(ai.status())

@app.route('/predict', methods=['POST'])
def predict():
    d = request.get_json()
    try: return jsonify(ai.predict(d['last_numbers']))
    except Exception as e: return jsonify({'error': str(e)}), 400

@app.route('/update', methods=['POST'])
def update():
    d = request.get_json(); ai.update(d['last_numbers'], d['true_number'])
    return jsonify({'status': 'ok'})

@app.route('/fetch_latest')
def fetch_latest():
    try:
        csv = "wingo_history.csv"
        nd = fetch_history(GAME_CODE, PAGE_SIZE, 2)
        if nd is None or nd.empty:
            if os.path.exists(csv):
                df = pd.read_csv(csv, dtype={'issueNumber': str})
                recent = df['number'].values[-LOOKBACK:].astype(int).tolist()
                hist = [{'issueNumber':str(r[0]),'number':int(r[1])} for r in df[['issueNumber','number']].values[-60:]]
                return jsonify({'last_numbers':recent,'total_records':len(df),'history':hist,'prediction':None,'note':'Using cached data'})
            return jsonify({'error':'API fetch failed and no cached data'})
        m = merge_csv(csv, nd)
        recent = m['number'].values[-LOOKBACK:].astype(int).tolist()
        hist = [{'issueNumber':str(r[0]),'number':int(r[1])} for r in m[['issueNumber','number']].values[-60:]]
        ai.total_records = len(m)
        pred = ai.predict(recent) if len(recent) >= LOOKBACK and ai.ready else None
        log(f"Fetched {len(m)} records", "info")
        return jsonify({'last_numbers':recent,'total_records':len(m),'history':hist,'prediction':pred})
    except Exception as e: return jsonify({'error':str(e)}), 500

@app.route('/history')
def history():
    csv = "wingo_history.csv"
    if os.path.exists(csv):
        df = pd.read_csv(csv, dtype={'issueNumber':str})
        h = [{'issueNumber':str(r[0]),'number':int(r[1])} for r in df[['issueNumber','number']].values[-60:]]
        return jsonify({'history':h,'total':len(df)})
    return jsonify({'history':[],'total':0})

@app.route('/train', methods=['POST'])
def train_route():
    def go():
        try:
            csv = ensure_csv_ready()
            if csv is None:
                log("No data available","error"); return
            ai.train_all(csv)
        except Exception as e:
            log(f"Train error: {e}","error")
    threading.Thread(target=go, daemon=True).start()
    return jsonify({'status':'started'})

@app.route('/quick_check', methods=['POST'])
def quick_check():
    d = request.get_json()
    try:
        result = ai.quick_check(d['last_numbers'], d['actual_number'])
        if result is None: return jsonify({'error':'Not ready'}), 400
        return jsonify(result)
    except Exception as e: return jsonify({'error':str(e)}), 500

@app.route('/live', methods=['POST'])
def live_route():
    if ai.is_live: return jsonify({'status':'running'})
    d = request.get_json() or {}
    interval = d.get('interval', 30)
    ai.live_interval = interval
    ai.is_live = True
    threading.Thread(target=live_loop, daemon=True).start()
    return jsonify({'status':'started', 'interval': interval})

@app.route('/live_stop', methods=['POST'])
def live_stop():
    ai.is_live = False
    log("Live STOPPED","info")
    return jsonify({'status':'stopped'})

@app.route('/last_result')
def last_result():
    csv = "wingo_history.csv"
    if os.path.exists(csv):
        df = pd.read_csv(csv, dtype={'issueNumber':str})
        if len(df) > 0:
            last = df.iloc[-1]
            return jsonify({"number": int(last['number']), "issue": str(last['issueNumber'])})
    return jsonify({"number": None, "issue": "--"})

def live_loop():
    log(f"Live ON! ({ai.live_interval}s polling)","success")
    csv = "wingo_history.csv"
    if not os.path.exists(f'{MODEL_DIR}/rf_opt.pkl'):
        csv_ready = ensure_csv_ready()
        if csv_ready: csv = csv_ready
        try:
            ai.train_all(csv)
        except Exception as e:
            log(f"Training error: {e}","error")
    last = None
    last_issue_num = None
    predicted_size = None
    predicted_seq = None
    last_actual = None
    while ai.is_live:
        try:
            if not os.path.exists(csv):
                time.sleep(ai.live_interval)
                continue
            nd = fetch_history(GAME_CODE, PAGE_SIZE, 2)
            if nd is not None and not nd.empty:
                m = merge_csv(csv, nd)
                rn = m['number'].values[-LOOKBACK:].astype(int).tolist()
                iss = m['issueNumber'].values[-1]
                actual = rn[-1]
                last_actual = {"number": actual, "issue": str(iss)}
                cur_issue_num = int(iss) if iss.isdigit() else 0
                if last is None or iss != last:
                    if predicted_size is not None and last_issue_num is not None:
                        actual_size = "big" if actual >= 5 else "small"
                        if predicted_size == "skip":
                            log(f"SKIPPED | Actual={actual}({actual_size})","info")
                        elif predicted_size == actual_size:
                            ai.stats["wins"] += 1
                            ai.stats["total"] += 1
                            ai.stats["accuracy"] = round((ai.stats["wins"]/ai.stats["total"])*100,1)
                            log(f"WIN! Pred={predicted_size} = Actual={actual_size} | Actual={actual}","success")
                        else:
                            ai.stats["losses"] += 1
                            ai.stats["total"] += 1
                            ai.stats["accuracy"] = round((ai.stats["wins"]/ai.stats["total"])*100,1)
                            log(f"LOSS! Pred={predicted_size} != Actual={actual_size} | Actual={actual}","error")
                    if len(rn) >= LOOKBACK and ai.ready:
                        p = ai.predict(rn)
                        if p.get("skip", False):
                            predicted_size = "skip"
                            log(f"SKIP ({p.get('agree_count',0)}/{p.get('total_votes',0)} agree)","info")
                        else:
                            predicted_size = p['size']
                            predicted_seq = rn
                            log(f"PREDICT: {p['size'].upper()} ({p.get('agree_count',0)}/{p.get('total_votes',0)} agree)","success")
                        last_issue_num = cur_issue_num
                    last = iss
            time.sleep(ai.live_interval)
        except Exception as e:
            log(f"Live error: {e}","error"); time.sleep(3)

if __name__ == '__main__':
    import sys
    port = int(os.environ.get('PORT', 5000))
    if os.path.exists(f'{MODEL_DIR}/rf_opt.pkl'):
        ai.load()
    else:
        log("No models found - will train on first request", "info")
    if len(sys.argv) > 1 and sys.argv[1] == '--live':
        ai.is_live = True
        threading.Thread(target=live_loop, daemon=True).start()
    print("\n" + "="*50)
    print("  WinGo AI Pro v4 - Ensemble (Freq+Pattern+RF)")
    print(f"  http://localhost:{port}")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
