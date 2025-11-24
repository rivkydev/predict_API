from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import json
import traceback
import re
import threading
import logging

# Machine Learning
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

# Configuration
import warnings
warnings.filterwarnings('ignore')

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Izinkan akses dari Web Builder (Cross-Origin Resource Sharing)

# ==================== HELPER: JSON ENCODER ====================
class NpEncoder(json.JSONEncoder):
    """Mengubah tipe data NumPy menjadi native Python agar bisa jadi JSON"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

app.json_encoder = NpEncoder

# ==================== KONFIGURASI ====================
class Config:
    WITA = timezone(timedelta(hours=8))
    RESET_HOUR = 8
    HEADERS = {
        "accept": "application/vnd.xenvelop+json",
        "x-language": "id_ID",
        "x-whence": "22",
        "x-referral": "1",
        "x-group": "1357",
        "x-bundleid": "org.xbet.client1",
        "appguid": "a162c8ff5632b2fe_2",
        "x-fcountry": "72",
        "content-type": "application/json; charset=utf-8",
        "user-agent": "org.xbet.client1-user-agent/1xbet-prod-149(25025)",
        "version": "1xbet-prod-149(25025)",
        "x-devicemanufacturer": "vivo",
        "x-devicemodel": "V2434",
        "x-country": "72",
        "cache-control": "no-cache",
        "accept-encoding": "br,gzip"
    }
    ML_CONFIDENCE_THRESHOLD = 0.65
    VALUE_BET_EDGE = 0.05
    MIN_MATCHES_FOR_ANALYSIS = 5

# ==================== CORE LOGIC CLASSES ====================
# (Logic ML dan API Manager dipertahankan, disesuaikan tanpa GUI)

class APIManager:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(Config.HEADERS)
    
    def get_historical_data(self):
        now = datetime.now(Config.WITA)
        date_to = now
        date_from = now - timedelta(days=1)
        date_from = date_from.replace(hour=Config.RESET_HOUR, minute=0, second=0, microsecond=0)
        
        timestamp_from = int(date_from.timestamp())
        timestamp_to = int(date_to.timestamp())
        
        url = f"https://5y7xvr1pm3q.com/ResultService/mobile/api/v2/games?ChampIds=2648573&country=72&DateFrom={timestamp_from}&DateTo={timestamp_to}&lng=id_ID&ref=1&gr=1357"
        
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching historical: {e}")
        return None

    def get_upcoming_matches(self):
        url = "https://5y7xvr1pm3q.com/MainFeedLive/mobile/v1/gamesByChamp?cfView=3&champIds=2648573&country=72&gr=1357&lng=id_ID&mode=2&ref=1&whence=22"
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                games = []
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'games' in item:
                            games.extend(item['games'])
                elif isinstance(data, dict) and 'data' in data:
                    if isinstance(data['data'], list):
                        for item in data['data']:
                            if isinstance(item, dict) and 'games' in item:
                                games.extend(item['games'])
                return games
        except Exception as e:
            logger.error(f"Error fetching upcoming: {e}")
        return []

class AdvancedMLEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}

    def create_ensemble_model(self):
        return VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('xgb', XGBClassifier(n_estimators=50, random_state=42)),
            ('lr', LogisticRegression(random_state=42))
        ], voting='soft')

    def prepare_features(self, team_stats, home_team, away_team):
        home_features = team_stats.get(home_team, {})
        away_features = team_stats.get(away_team, {})
        if not home_features or not away_features:
            return None
        
        features = [
            home_features.get('win_rate', 0),
            away_features.get('win_rate', 0),
            home_features.get('avg_goals_for', 0),
            away_features.get('avg_goals_for', 0),
            home_features.get('avg_goals_against', 0),
            away_features.get('avg_goals_against', 0),
            home_features.get('momentum', 0),
            away_features.get('momentum', 0),
        ]
        return np.array(features).reshape(1, -1)

    def train_models(self, X, y, model_type='result'):
        if len(X) < 10: return None
        if model_type not in self.models:
            self.models[model_type] = self.create_ensemble_model()
            self.scalers[model_type] = StandardScaler()
        
        X_scaled = self.scalers[model_type].fit_transform(X)
        self.models[model_type].fit(X_scaled, y)
        return self.models[model_type]

class AdvancedFC25Analyzer:
    def __init__(self, historical_data=None):
        self.data = historical_data
        self.df = None
        self.team_stats = defaultdict(lambda: self._create_default_stats())
        self.elo_ratings = {}
        self.streak_patterns = defaultdict(list)
        self.momentum_factors = defaultdict(float)
        self.ml_engine = AdvancedMLEngine()

    def _create_default_stats(self):
        return {
            'played': 0, 'won': 0, 'lost': 0, 'draw': 0,
            'goals_for': 0, 'goals_against': 0,
            'win_rate': 0, 'avg_goals_for': 0, 'avg_goals_against': 0,
            'momentum': 0, 'recent_form': [],
            'home_record': {'played': 0, 'won': 0},
            'away_record': {'played': 0, 'won': 0}
        }

    def analyze_with_ml(self):
        self.analyze()
        X, y = self._prepare_ml_data()
        if X is not None and len(X) > 10:
            self.ml_engine.train_models(X, y, 'result')
            logger.info("ML Models trained successfully")

    def analyze(self):
        if self.data and 'data' in self.data and 'items' in self.data['data']:
            items = self.data['data']['items']
            self.df = pd.DataFrame(items)
            self._preprocess_data()
            self._calculate_elo_ratings()
            self._analyze_patterns()
            self._analyze_momentum()
            self._calculate_advanced_metrics()
        return self

    def _prepare_ml_data(self):
        if self.df is None or len(self.df) < 10: return None, None
        features_list = []
        targets = []
        for _, row in self.df.iterrows():
            home_team = row['opp1']
            away_team = row['opp2']
            feature_vector = self.ml_engine.prepare_features(self.team_stats, home_team, away_team)
            if feature_vector is not None:
                features_list.append(feature_vector[0])
                if row['score1'] > row['score2']: targets.append('home_win')
                elif row['score1'] < row['score2']: targets.append('away_win')
                else: targets.append('draw')
        return np.array(features_list), np.array(targets)

    def _preprocess_data(self):
        try:
            self.df[['score1', 'score2']] = self.df['scoreBase'].str.split(':', expand=True).astype(int)
            self.df['total_goals'] = self.df['score1'] + self.df['score2']
            self.df['winner'] = self.df.apply(lambda x: 'home' if x['score1'] > x['score2'] else ('away' if x['score2'] > x['score1'] else 'draw'), axis=1)
        except Exception: pass

    def _calculate_elo_ratings(self):
        for _, row in self.df.iterrows():
            home, away = row['opp1'], row['opp2']
            if home not in self.elo_ratings: self.elo_ratings[home] = 1500
            if away not in self.elo_ratings: self.elo_ratings[away] = 1500
            
            result = 'win' if row['winner'] == 'home' else ('loss' if row['winner'] == 'away' else 'draw')
            new_home, new_away = self._calculate_elo_rating(self.elo_ratings, home, away, result)
            self.elo_ratings[home] = new_home
            self.elo_ratings[away] = new_away

    def _calculate_elo_rating(self, team_elos, team1, team2, result, k_factor=32):
        rating1 = team_elos.get(team1, 1500)
        rating2 = team_elos.get(team2, 1500)
        expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
        actual = 1 if result == 'win' else (0.5 if result == 'draw' else 0)
        new_rating1 = rating1 + k_factor * (actual - expected1)
        new_rating2 = rating2 + k_factor * ((1 - actual) - (1 - expected1))
        return new_rating1, new_rating2

    def _analyze_patterns(self):
        for _, row in self.df.iterrows():
            home, away = row['opp1'], row['opp2']
            for team, score, opp_score, is_home in [(home, row['score1'], row['score2'], True), (away, row['score2'], row['score1'], False)]:
                stats = self.team_stats[team]
                stats['played'] += 1
                stats['goals_for'] += score
                stats['goals_against'] += opp_score
                if score > opp_score:
                    stats['won'] += 1
                    stats['recent_form'].append('W')
                    self.streak_patterns[team].append(1)
                elif score < opp_score:
                    stats['lost'] += 1
                    stats['recent_form'].append('L')
                    self.streak_patterns[team].append(-1)
                else:
                    stats['draw'] += 1
                    stats['recent_form'].append('D')
                    self.streak_patterns[team].append(0)
                
                if len(stats['recent_form']) > 10: stats['recent_form'] = stats['recent_form'][-10:]
                if len(self.streak_patterns[team]) > 10: self.streak_patterns[team] = self.streak_patterns[team][-10:]
                
                if is_home:
                    stats['home_record']['played'] += 1
                    if score > opp_score: stats['home_record']['won'] += 1
                else:
                    stats['away_record']['played'] += 1
                    if score > opp_score: stats['away_record']['won'] += 1

    def _analyze_momentum(self):
        for team, sequence in self.streak_patterns.items():
            if len(sequence) >= 5:
                weights = np.linspace(0.5, 1.0, len(sequence))
                momentum = np.average(sequence, weights=weights)
                self.momentum_factors[team] = momentum
                self.team_stats[team]['momentum'] = momentum

    def _calculate_advanced_metrics(self):
        for _, stats in self.team_stats.items():
            if stats['played'] > 0:
                stats['win_rate'] = stats['won'] / stats['played']
                stats['avg_goals_for'] = stats['goals_for'] / stats['played']
                stats['avg_goals_against'] = stats['goals_against'] / stats['played']
    
    def find_team_in_data(self, team_name):
        if team_name in self.team_stats: return team_name
        return None

class MLBettingPredictor:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.confidence_threshold = Config.ML_CONFIDENCE_THRESHOLD
        self.value_bet_edge = Config.VALUE_BET_EDGE

    def extract_odds(self, match_data):
        odds = {'1': None, 'X': None, '2': None, 'totals': {}}
        try:
            for group in match_data.get('eventGroups', []):
                if group.get('groupId') == 1 and len(group.get('events', [])) >= 3:
                    events = group['events']
                    if events[0]: odds['1'] = events[0][0].get('cf')
                    if events[1]: odds['X'] = events[1][0].get('cf')
                    if events[2]: odds['2'] = events[2][0].get('cf')
                elif group.get('groupId') == 4:
                    for event_list in group.get('events', []):
                        if event_list:
                            event = event_list[0]
                            if event.get('type') == 9: odds['totals'][f"over_{event.get('parameter')}"] = event.get('cf')
                            elif event.get('type') == 10: odds['totals'][f"under_{event.get('parameter')}"] = event.get('cf')
        except: pass
        return odds

    def predict_match(self, home_team, away_team, match_data=None):
        hist_home = self.analyzer.find_team_in_data(home_team)
        hist_away = self.analyzer.find_team_in_data(away_team)
        
        if not hist_home or not hist_away: return None
        
        home_stats = self.analyzer.team_stats[hist_home]
        away_stats = self.analyzer.team_stats[hist_away]
        
        if home_stats['played'] < Config.MIN_MATCHES_FOR_ANALYSIS or away_stats['played'] < Config.MIN_MATCHES_FOR_ANALYSIS:
            return None

        odds = self.extract_odds(match_data) if match_data else {}
        
        # ML Prediction
        features = self.analyzer.ml_engine.prepare_features(self.analyzer.team_stats, home_team, away_team)
        ml_probs = None
        if features is not None:
            model = self.analyzer.ml_engine.models.get('result')
            scaler = self.analyzer.ml_engine.scalers.get('result')
            if model and scaler:
                ml_probs = model.predict_proba(scaler.transform(features))[0]

        # Calculate Probabilities
        home_elo = self.analyzer.elo_ratings.get(home_team, 1500)
        away_elo = self.analyzer.elo_ratings.get(away_team, 1500)
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        
        if ml_probs is not None:
            home_prob = (expected_home * 0.4 + home_stats['win_rate'] * 0.3 + ml_probs[0] * 0.3)
            away_prob = ((1 - expected_home) * 0.4 + away_stats['win_rate'] * 0.3 + ml_probs[2] * 0.3)
        else:
            home_prob = (expected_home * 0.6 + home_stats['win_rate'] * 0.4)
            away_prob = ((1 - expected_home) * 0.6 + away_stats['win_rate'] * 0.4)

        predictions = []
        
        # Helper to calc value
        def get_value(prob, odd_val):
            if not odd_val or odd_val <= 1: return 0, False
            edge = prob - (1/odd_val)
            return edge * 100, edge > self.value_bet_edge

        # Check Home
        val, is_val = get_value(home_prob, odds.get('1'))
        if is_val and home_prob > self.confidence_threshold:
            predictions.append(self._make_pred_obj('1X2', '1 (Home Win)', home_prob, odds['1'], val, home_elo, away_elo))

        # Check Away
        val, is_val = get_value(away_prob, odds.get('2'))
        if is_val and away_prob > self.confidence_threshold:
            predictions.append(self._make_pred_obj('1X2', '2 (Away Win)', away_prob, odds['2'], val, away_elo, home_elo))

        # Check Over Goals
        expected_total = home_stats['avg_goals_for'] + away_stats['avg_goals_for']
        for k, v in odds.get('totals', {}).items():
            if 'over' in k:
                line = float(k.replace('over_', ''))
                over_prob = min(0.9, (expected_total / line) * 0.8) if line >= 10 else min(0.8, (expected_total / line) * 0.7)
                val, is_val = get_value(over_prob, v)
                if is_val and over_prob > 0.5:
                    predictions.append({
                        'market': 'Total Goals',
                        'pick': f'Over {line}',
                        'probability': round(over_prob * 100, 1),
                        'odds': v,
                        'value_edge': round(val, 1),
                        'reasoning': f"Exp Goals: {expected_total:.1f}",
                        'strength': 'HIGH' if val > 8 else 'MEDIUM'
                    })

        predictions.sort(key=lambda x: x['value_edge'], reverse=True)
        return predictions

    def _make_pred_obj(self, market, pick, prob, odds, val, elo1, elo2):
        return {
            'market': market,
            'pick': pick,
            'probability': round(prob * 100, 1),
            'odds': odds,
            'value_edge': round(val, 1),
            'reasoning': f"ELO Diff: {elo1 - elo2:.0f}",
            'strength': 'HIGH' if val > 10 else 'MEDIUM'
        }

def normalize_team_name(name):
    if not name: return ""
    name = str(name).lower().strip()
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\b(c\.?f\.?|f\.?c\.?|club|team|fc)\b', '', name)
    return name.strip()

# ==================== STATE MANAGEMENT ====================
class SystemState:
    def __init__(self):
        self.analyzer = None
        self.predictor = None
        self.is_loading = False
        self.last_updated = None

state = SystemState()

# ==================== API ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Cek status server"""
    return jsonify({
        "status": "online",
        "ml_ready": state.predictor is not None,
        "last_updated": state.last_updated,
        "is_loading": state.is_loading
    })

@app.route('/refresh', methods=['POST'])
def refresh_data():
    """Trigger update data manual"""
    if state.is_loading:
        return jsonify({"status": "error", "message": "Already loading data"}), 429

    def load_task():
        state.is_loading = True
        logger.info("Starting data refresh...")
        api = APIManager()
        hist_data = api.get_historical_data()
        
        if hist_data:
            analyzer = AdvancedFC25Analyzer(hist_data)
            analyzer.analyze_with_ml()
            state.analyzer = analyzer
            state.predictor = MLBettingPredictor(analyzer)
            state.last_updated = datetime.now().isoformat()
            logger.info("Data refresh complete.")
        else:
            logger.error("Failed to load data.")
        
        state.is_loading = False

    thread = threading.Thread(target=load_task)
    thread.start()
    
    return jsonify({"status": "accepted", "message": "Data refresh started in background"})

@app.route('/teams', methods=['GET'])
def get_teams():
    """Dapatkan list semua tim"""
    if not state.analyzer:
        return jsonify({"error": "System not initialized. Call /refresh first."}), 503
    
    teams = []
    for name, stats in state.analyzer.team_stats.items():
        teams.append({
            "name": name,
            "played": stats['played'],
            "win_rate": round(stats['win_rate'], 2),
            "elo": round(state.analyzer.elo_ratings.get(name, 1500))
        })
    
    # Sort by Elo
    teams.sort(key=lambda x: x['elo'], reverse=True)
    return jsonify({"count": len(teams), "teams": teams})

@app.route('/predictions', methods=['GET'])
def get_predictions():
    """Dapatkan prediksi match yang akan datang"""
    if not state.predictor:
        return jsonify({"error": "Model not ready. Call /refresh first."}), 503

    api = APIManager()
    upcoming = api.get_upcoming_matches()
    
    results = []
    for match in upcoming:
        home = match.get('opponent1', {}).get('fullName', 'Unknown')
        away = match.get('opponent2', {}).get('fullName', 'Unknown')
        league = match.get('champ', {}).get('name', 'Unknown')
        
        preds = state.predictor.predict_match(home, away, match)
        
        if preds:
            results.append({
                "match_id": match.get('id'),
                "home_team": home,
                "away_team": away,
                "league": league,
                "start_date": match.get('start', 0),
                "predictions": preds
            })
    
    # Filter hanya yang punya prediksi value bet
    results = [r for r in results if len(r['predictions']) > 0]
    results.sort(key=lambda x: x['predictions'][0]['value_edge'], reverse=True)
    
    return json.dumps({"count": len(results), "matches": results}, cls=NpEncoder)

@app.route('/analyze/team/<team_name>', methods=['GET'])
def analyze_team(team_name):
    """Detail analisis tim tertentu"""
    if not state.analyzer:
        return jsonify({"error": "Not initialized"}), 503
    
    # Simple fuzzy search
    target = None
    if team_name in state.analyzer.team_stats:
        target = team_name
    else:
        norm_input = normalize_team_name(team_name)
        for real_name in state.analyzer.team_stats.keys():
            if normalize_team_name(real_name) == norm_input:
                target = real_name
                break
    
    if not target:
        return jsonify({"error": "Team not found"}), 404
        
    stats = state.analyzer.team_stats[target]
    elo = state.analyzer.elo_ratings.get(target, 1500)
    
    return json.dumps({
        "name": target,
        "elo": elo,
        "stats": stats,
        "momentum": state.analyzer.momentum_factors.get(target, 0)
    }, cls=NpEncoder)

if __name__ == '__main__':
    # Untuk development run
    app.run(host='0.0.0.0', port=5000, debug=True)