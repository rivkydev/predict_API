# fc25_api.py
from flask import Flask, jsonify, request
from flask_restx import Api, Resource, fields
from flask_cors import CORS
import threading
import time
import json
from datetime import datetime
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict
import json
import traceback
import re
import time
import threading
from typing import Dict, List, Optional, Tuple

# Machine Learning & Data Science
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from scipy import stats
import joblib

def normalize_team_name(name):
    """Normalisasi nama tim untuk matching yang lebih baik"""
    if not name:
        return ""
    name = str(name).lower().strip()
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\b(c\.?f\.?|f\.?c\.?|club|team|fc)\b', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip()

# ==================== KONFIGURASI LANJUT ====================
class Config:
    WITA = timezone(timedelta(hours=8))
    RESET_HOUR = 8
    
    # API Configuration
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
    
    # Model Parameters
    ML_CONFIDENCE_THRESHOLD = 0.65
    VALUE_BET_EDGE = 0.05
    MIN_MATCHES_FOR_ANALYSIS = 5

# ==================== DEBUG & LOGGING SYSTEM ====================
class DebugLogger:
    def __init__(self):
        self.log_file = "fc25_debug.log"
    
    def log_api_call(self, url, response, error=None):
        """Log detail API call dengan handling timedelta"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Handle timedelta serialization
        response_time = None
        if hasattr(response, 'elapsed'):
            try:
                response_time = float(response.elapsed.total_seconds())
            except:
                response_time = None
        
        log_entry = {
            'timestamp': timestamp,
            'url': url,
            'status_code': getattr(response, 'status_code', 'NO_RESPONSE'),
            'response_time': response_time,
            'error': str(error) if error else None,
            'response_preview': str(getattr(response, 'text', ''))[:200] if hasattr(response, 'text') else 'NO_RESPONSE'
        }
        
        print(f"🔍 API CALL: {json.dumps(log_entry, indent=2)}")
        
        # Save to debug file
        try:
            with open("api_debug.json", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"❌ Error saving debug log: {e}")
    
    def log_data_structure(self, data, label):
        """Log struktur data untuk debugging"""
        if data is None:
            print(f"{label}: NULL DATA")
            return
        
        if isinstance(data, dict):
            print(f"{label} KEYS: {list(data.keys())}")
            if 'data' in data and isinstance(data['data'], dict):
                print(f"{label} DATA KEYS: {list(data['data'].keys())}")
                if 'items' in data['data']:
                    print(f"{label} ITEMS COUNT: {len(data['data']['items'])}")
        elif isinstance(data, list):
            print(f"{label} LENGTH: {len(data)}")
            if len(data) > 0:
                print(f"{label} FIRST ITEM TYPE: {type(data[0])}")

# ==================== API MANAGER ====================
class APIManager:
    def __init__(self):
        self.debug = DebugLogger()
        self.session = requests.Session()
        self.session.headers.update(Config.HEADERS)
    
    def get_historical_data(self):
        """Ambil data historis: dari 08:00 WITA kemarin sampai sekarang"""
        now = datetime.now(Config.WITA)
        
        # Date TO = jam laptop sekarang
        date_to = now
        
        # Date FROM = 08:00 WITA kemarin
        date_from = now - timedelta(days=1)
        date_from = date_from.replace(hour=Config.RESET_HOUR, minute=0, second=0, microsecond=0)
        
        # Convert to timestamps
        timestamp_from = int(date_from.timestamp())
        timestamp_to = int(date_to.timestamp())
        
        print(f"📥 Fetching historical data...")
        print(f"   FROM: {date_from.strftime('%d/%m/%Y %H:%M WITA')}")
        print(f"   TO:   {date_to.strftime('%d/%m/%Y %H:%M WITA')}")
        
        url = f"https://5y7xvr1pm3q.com/ResultService/mobile/api/v2/games?ChampIds=2648573&country=72&DateFrom={timestamp_from}&DateTo={timestamp_to}&lng=id_ID&ref=1&gr=1357"
        
        try:
            response = self.session.get(url, timeout=30)
            self.debug.log_api_call(url, response, None)
            
            if response.status_code == 200:
                data = response.json()
                self.debug.log_data_structure(data, "HISTORICAL_DATA")
                
                if 'data' in data and 'items' in data['data']:
                    items = data['data']['items']
                    print(f"   ✅ Got {len(items)} matches")
                    return data
                else:
                    print(f"   ⚠️  No items in response structure")
            else:
                print(f"   ❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        return None

    def get_upcoming_matches(self):
        """Ambil pertandingan akan datang dengan ODDS"""
        url = "https://5y7xvr1pm3q.com/MainFeedLive/mobile/v1/gamesByChamp?cfView=3&champIds=2648573&country=72&gr=1357&lng=id_ID&mode=2&ref=1&whence=22"
        
        print(f"📥 Fetching upcoming matches...")
        
        try:
            response = self.session.get(url, timeout=30)
            self.debug.log_api_call(url, response, None)
            
            if response.status_code == 200:
                data = response.json()
                self.debug.log_data_structure(data, "UPCOMING_DATA")
                
                # Extract games from nested structure
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
                
                print(f"   ✅ Total games extracted: {len(games)}")
                return games
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        return []

# ==================== ADVANCED ML ENGINE ====================
class AdvancedMLEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def create_ensemble_model(self):
        """Membuat ensemble model dengan berbagai algoritma"""
        return VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('xgb', XGBClassifier(n_estimators=50, random_state=42)),
            ('lr', LogisticRegression(random_state=42))
        ], voting='soft')
    
    def prepare_features(self, team_stats, home_team, away_team):
        """Mempersiapkan features untuk ML model"""
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
        """Train model dengan cross-validation"""
        if len(X) < 10:
            print("⚠️  Not enough data for ML training")
            return None
            
        if model_type not in self.models:
            self.models[model_type] = self.create_ensemble_model()
            self.scalers[model_type] = StandardScaler()
            
        # Scale features
        X_scaled = self.scalers[model_type].fit_transform(X)
        
        # Train model
        self.models[model_type].fit(X_scaled, y)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.models[model_type], X_scaled, y, cv=3)
        print(f"🤖 ML Model {model_type} - CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return self.models[model_type]

# ==================== ADVANCED ANALYZER ====================
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
        """Analisis dengan integrasi ML"""
        print("🤖 Starting ML-enhanced analysis...")
        self.analyze()  # Basic analysis first
        
        # Prepare data for ML
        X, y = self._prepare_ml_data()
        if X is not None and len(X) > 10:
            self.ml_engine.train_models(X, y, 'result')
            print("✅ ML Models trained successfully")
        else:
            print("⚠️  Not enough data for ML training")
    
    def _prepare_ml_data(self):
        """Prepare data untuk training ML"""
        if self.df is None or len(self.df) < 10:
            return None, None
        
        features_list = []
        targets = []
        
        for _, row in self.df.iterrows():
            home_team = row['opp1']
            away_team = row['opp2']
            
            feature_vector = self.ml_engine.prepare_features(self.team_stats, home_team, away_team)
            if feature_vector is not None:
                features_list.append(feature_vector[0])
                
                # Determine target
                if row['score1'] > row['score2']:
                    targets.append('home_win')
                elif row['score1'] < row['score2']:
                    targets.append('away_win')
                else:
                    targets.append('draw')
        
        print(f"📊 Prepared {len(features_list)} samples for ML training")
        return np.array(features_list), np.array(targets)

    def analyze(self):
        """Analisis data historis dengan teknik advanced"""
        try:
            if self.data and 'data' in self.data and 'items' in self.data['data']:
                items = self.data['data']['items']
                print(f"📊 Analyzing {len(items)} matches with advanced algorithms...")
                
                self.df = pd.DataFrame(items)
                
                self._preprocess_data()
                self._calculate_elo_ratings()
                self._analyze_patterns()
                self._analyze_momentum()
                self._calculate_advanced_metrics()
                
                print(f"✅ Analysis complete: {len(self.team_stats)} teams analyzed")
                return self
                
            else:
                print("❌ No valid historical data")
                return self
                
        except Exception as e:
            print(f"❌ Error in analysis: {e}")
            traceback.print_exc()
            return self
    
    def _calculate_advanced_metrics(self):
        """Hitung metrics advanced untuk setiap tim"""
        for team, stats in self.team_stats.items():
            if stats['played'] > 0:
                stats['win_rate'] = stats['won'] / stats['played']
                stats['avg_goals_for'] = stats['goals_for'] / stats['played']
                stats['avg_goals_against'] = stats['goals_against'] / stats['played']
    
    def _preprocess_data(self):
        """Pre-process data"""
        try:
            # Split scoreBase
            self.df[['score1', 'score2']] = self.df['scoreBase'].str.split(':', expand=True).astype(int)
            self.df['total_goals'] = self.df['score1'] + self.df['score2']
            self.df['winner'] = self.df.apply(
                lambda x: 'home' if x['score1'] > x['score2'] else 
                ('away' if x['score2'] > x['score1'] else 'draw'), axis=1)
            
            print(f"✅ Data preprocessing complete")
            
        except Exception as e:
            print(f"❌ Error in preprocessing: {e}")
    
    def _calculate_elo_ratings(self):
        """Hitung ELO rating untuk setiap tim"""
        print("🎯 Calculating ELO ratings...")
        
        for _, row in self.df.iterrows():
            home = row['opp1']
            away = row['opp2']
            
            if home not in self.elo_ratings:
                self.elo_ratings[home] = 1500
            if away not in self.elo_ratings:
                self.elo_ratings[away] = 1500
            
            if row['winner'] == 'home':
                result_home = 'win'
            elif row['winner'] == 'away':
                result_home = 'loss'
            else:
                result_home = 'draw'
            
            new_home, new_away = self._calculate_elo_rating(
                self.elo_ratings, home, away, result_home
            )
            
            self.elo_ratings[home] = new_home
            self.elo_ratings[away] = new_away
    
    def _calculate_elo_rating(self, team_elos, team1, team2, result, k_factor=32):
        """Sistem ELO Rating"""
        rating1 = team_elos.get(team1, 1500)
        rating2 = team_elos.get(team2, 1500)
        
        expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
        
        if result == 'win':
            actual = 1
        elif result == 'draw':
            actual = 0.5
        else:
            actual = 0
        
        new_rating1 = rating1 + k_factor * (actual - expected1)
        new_rating2 = rating2 + k_factor * ((1 - actual) - (1 - expected1))
        
        return new_rating1, new_rating2
    
    def _analyze_patterns(self):
        """Analisis pola dan trend"""
        print("🔍 Analyzing patterns and trends...")
        
        for _, row in self.df.iterrows():
            home = row['opp1']
            away = row['opp2']
            
            for team, score, opp_score, is_home in [
                (home, row['score1'], row['score2'], True),
                (away, row['score2'], row['score1'], False)
            ]:
                stats = self.team_stats[team]
                stats['played'] += 1
                stats['goals_for'] += score
                stats['goals_against'] += opp_score
                
                # Update sequences
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
                
                # Keep recent form (last 10 matches)
                if len(stats['recent_form']) > 10:
                    stats['recent_form'] = stats['recent_form'][-10:]
                if len(self.streak_patterns[team]) > 10:
                    self.streak_patterns[team] = self.streak_patterns[team][-10:]
                
                # Home/Away stats
                if is_home:
                    stats['home_record']['played'] += 1
                    if score > opp_score:
                        stats['home_record']['won'] += 1
                else:
                    stats['away_record']['played'] += 1
                    if score > opp_score:
                        stats['away_record']['won'] += 1
    
    def _analyze_momentum(self):
        """Analisis momentum tim"""
        print("📈 Calculating momentum factors...")
        
        for team, sequence in self.streak_patterns.items():
            if len(sequence) >= 5:
                # Weighted recent form
                weights = np.linspace(0.5, 1.0, len(sequence))
                momentum = np.average(sequence, weights=weights)
                self.momentum_factors[team] = momentum
                self.team_stats[team]['momentum'] = momentum

    def find_team_in_data(self, team_name):
        """Cari tim dalam data"""
        if team_name in self.team_stats:
            return team_name
        
        normalized = normalize_team_name(team_name)
        for hist_team in self.team_stats.keys():
            if normalize_team_name(hist_team) == normalized:
                return hist_team
        
        return None

# ==================== ADVANCED PREDICTOR ====================
class MLBettingPredictor:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.confidence_threshold = Config.ML_CONFIDENCE_THRESHOLD
        self.value_bet_edge = Config.VALUE_BET_EDGE
    
    def extract_odds(self, match_data):
        """Extract odds dari match data"""
        odds = {
            '1': None, 'X': None, '2': None, 
            'totals': {}
        }
        
        try:
            event_groups = match_data.get('eventGroups', [])
            
            for group in event_groups:
                group_id = group.get('groupId')
                events = group.get('events', [])
                
                # Match Result (1X2)
                if group_id == 1:
                    if len(events) >= 3:
                        if len(events[0]) > 0 and 'cf' in events[0][0]:
                            odds['1'] = events[0][0]['cf']
                        if len(events[1]) > 0 and 'cf' in events[1][0]:
                            odds['X'] = events[1][0]['cf']
                        if len(events[2]) > 0 and 'cf' in events[2][0]:
                            odds['2'] = events[2][0]['cf']
                
                # Total Goals
                elif group_id == 4:
                    for event_list in events:
                        if len(event_list) > 0:
                            event = event_list[0]
                            param = event.get('parameter')
                            event_type = event.get('type')
                            cf = event.get('cf')
                            
                            if param and cf:
                                if event_type == 9:  # Over
                                    odds['totals'][f'over_{param}'] = cf
                                elif event_type == 10:  # Under
                                    odds['totals'][f'under_{param}'] = cf
                    
        except Exception as e:
            print(f"   ⚠️  Error parsing odds: {e}")
            
        return odds
    
    def calculate_value_bet(self, prob, odds):
        """Hitung value betting"""
        if not odds or odds <= 1:
            return 0, False
        
        implied_prob = 1 / odds
        edge = prob - implied_prob
        
        # Value bet jika edge > threshold
        is_value = edge > self.value_bet_edge
        value_score = edge * 100
        
        return value_score, is_value
    
    def predict_with_ml(self, home_team, away_team):
        """Prediksi menggunakan ML model"""
        features = self.analyzer.ml_engine.prepare_features(
            self.analyzer.team_stats, home_team, away_team
        )
        
        if features is None:
            return None
        
        model = self.analyzer.ml_engine.models.get('result')
        scaler = self.analyzer.ml_engine.scalers.get('result')
        
        if model and scaler:
            features_scaled = scaler.transform(features)
            probabilities = model.predict_proba(features_scaled)[0]
            
            return {
                'home_win_prob': probabilities[0],
                'draw_prob': probabilities[1],
                'away_win_prob': probabilities[2]
            }
        
        return None
    
    def predict_match(self, home_team, away_team, match_data=None):
        """Prediksi comprehensive dengan ML dan statistical analysis"""
        print(f"\n{'='*60}")
        print(f"🎯 ANALYZING: {home_team} vs {away_team}")
        print(f"{'='*60}")
        
        # Find teams in historical data
        hist_home = self.analyzer.find_team_in_data(home_team)
        hist_away = self.analyzer.find_team_in_data(away_team)
        
        if not hist_home or not hist_away:
            print(f"❌ Insufficient historical data")
            return None
        
        home_stats = self.analyzer.team_stats[hist_home]
        away_stats = self.analyzer.team_stats[hist_away]
        
        if home_stats['played'] < Config.MIN_MATCHES_FOR_ANALYSIS or away_stats['played'] < Config.MIN_MATCHES_FOR_ANALYSIS:
            print(f"⚠️  Not enough matches (Home: {home_stats['played']}, Away: {away_stats['played']})")
            return None
        
        # Extract odds
        odds = self.extract_odds(match_data) if match_data else {}
        
        # ML Prediction
        ml_prediction = self.predict_with_ml(home_team, away_team)
        
        predictions = []
        
        # 1X2 Market Prediction
        home_elo = self.analyzer.elo_ratings.get(home_team, 1500)
        away_elo = self.analyzer.elo_ratings.get(away_team, 1500)
        
        elo_diff = home_elo - away_elo
        expected_home_win = 1 / (1 + 10 ** (-elo_diff / 400))
        
        # Combine statistical and ML probabilities
        if ml_prediction:
            home_prob = (expected_home_win * 0.4 + home_stats['win_rate'] * 0.3 + 
                        ml_prediction['home_win_prob'] * 0.3)
            away_prob = ((1 - expected_home_win) * 0.4 + away_stats['win_rate'] * 0.3 + 
                        ml_prediction['away_win_prob'] * 0.3)
        else:
            home_prob = (expected_home_win * 0.6 + home_stats['win_rate'] * 0.4)
            away_prob = ((1 - expected_home_win) * 0.6 + away_stats['win_rate'] * 0.4)
        
        # Format odds display
        odds_display = []
        if odds.get('1'): odds_display.append(f"1={odds['1']}")
        if odds.get('X'): odds_display.append(f"X={odds['X']}")
        if odds.get('2'): odds_display.append(f"2={odds['2']}")
        print(f"💰 ODDS: {' '.join(odds_display) if odds_display else 'N/A'}")
        
        # Home Win
        if home_prob > self.confidence_threshold and odds.get('1'):
            value, is_value = self.calculate_value_bet(home_prob, odds['1'])
            if is_value:
                confidence = min(90, home_prob * 100)
                predictions.append({
                    'market': '1X2',
                    'pick': '1 (Home Win)',
                    'probability': f"{home_prob:.1%}",
                    'confidence': f"{confidence:.0f}%",
                    'odds': odds['1'],
                    'value_edge': f"+{value:.1f}%",
                    'reasoning': f"ELO: {home_elo:.0f} vs {away_elo:.0f} | Win Rate: {home_stats['win_rate']:.1%}",
                    'strength': '🔥 VALUE BET' if value > 10 else '💪 STRONG'
                })
        
        # Away Win
        if away_prob > self.confidence_threshold and odds.get('2'):
            value, is_value = self.calculate_value_bet(away_prob, odds['2'])
            if is_value:
                confidence = min(90, away_prob * 100)
                predictions.append({
                    'market': '1X2',
                    'pick': '2 (Away Win)',
                    'probability': f"{away_prob:.1%}",
                    'confidence': f"{confidence:.0f}%",
                    'odds': odds['2'],
                    'value_edge': f"+{value:.1f}%",
                    'reasoning': f"ELO: {away_elo:.0f} vs {home_elo:.0f} | Win Rate: {away_stats['win_rate']:.1%}",
                    'strength': '🔥 VALUE BET' if value > 10 else '💪 STRONG'
                })
        
        # Total Goals Prediction
        home_avg = home_stats['avg_goals_for']
        away_avg = away_stats['avg_goals_for']
        expected_total = home_avg + away_avg
        
        totals = odds.get('totals', {})
        
        for total_key, total_odds in totals.items():
            if not total_key.startswith('over_'):
                continue
                
            line = float(total_key.replace('over_', ''))
            
            # Hitung probability untuk over
            if line >= 10:
                over_prob = min(0.9, (expected_total / line) * 0.8)
            else:
                over_prob = min(0.8, (expected_total / line) * 0.7)
            
            # Check value bet
            if over_prob > 0.45:
                value, is_value = self.calculate_value_bet(over_prob, total_odds)
                if is_value:
                    confidence = min(85, over_prob * 100)
                    predictions.append({
                        'market': 'Total Goals',
                        'pick': f'Over {line}',
                        'probability': f"{over_prob:.1%}",
                        'confidence': f"{confidence:.0f}%",
                        'odds': total_odds,
                        'value_edge': f"+{value:.1f}%",
                        'reasoning': f"Expected: {expected_total:.1f} goals | Line: {line}",
                        'strength': '🔥 VALUE BET' if value > 8 else '💪 STRONG'
                    })
        
        # Sort by value edge
        predictions.sort(key=lambda x: float(x['value_edge'].replace('+', '').replace('%', '')), reverse=True)
        
        return predictions if predictions else None

app = Flask(__name__)
CORS(app)
api = Api(app, 
          version='1.0', 
          title='FC25 Prediction API',
          description='Advanced Football Prediction API for FC25',
          doc='/docs')

# Namespaces
ns_matches = api.namespace('matches', description='Match operations')
ns_predictions = api.namespace('predictions', description='Prediction operations')
ns_analysis = api.namespace('analysis', description='Analysis operations')

# Global variables
api_manager = APIManager()
analyzer = None
predictor = None
last_update = None
is_analyzing = False

# Request models
prediction_request = api.model('PredictionRequest', {
    'home_team': fields.String(required=True, description='Home team name'),
    'away_team': fields.String(required=True, description='Away team name'),
    'match_data': fields.Raw(required=False, description='Optional match data with odds')
})

team_analysis_request = api.model('TeamAnalysisRequest', {
    'team_name': fields.String(required=True, description='Team name to analyze')
})

# Response models
prediction_response = api.model('Prediction', {
    'market': fields.String,
    'pick': fields.String,
    'probability': fields.String,
    'confidence': fields.String,
    'odds': fields.Float,
    'value_edge': fields.String,
    'reasoning': fields.String,
    'strength': fields.String
})

team_stats_response = api.model('TeamStats', {
    'team': fields.String,
    'played': fields.Integer,
    'won': fields.Integer,
    'lost': fields.Integer,
    'draw': fields.Integer,
    'win_rate': fields.Float,
    'avg_goals_for': fields.Float,
    'avg_goals_against': fields.Float,
    'momentum': fields.Float,
    'elo_rating': fields.Float,
    'recent_form': fields.List(fields.String)
})

@ns_matches.route('/historical')
class HistoricalMatches(Resource):
    def get(self):
        """Get historical match data"""
        try:
            data = api_manager.get_historical_data()
            if data:
                return {
                    'status': 'success',
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'status': 'error', 'message': 'Failed to fetch historical data'}, 500
        except Exception as e:
            return {'status': 'error', 'message': str(e)}, 500

@ns_matches.route('/upcoming')
class UpcomingMatches(Resource):
    def get(self):
        """Get upcoming matches with odds"""
        try:
            matches = api_manager.get_upcoming_matches()
            return {
                'status': 'success',
                'matches': matches,
                'count': len(matches),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}, 500

@ns_analysis.route('/analyze')
class AnalyzeData(Resource):
    def post(self):
        """Trigger data analysis and ML model training"""
        global analyzer, predictor, is_analyzing, last_update
        
        if is_analyzing:
            return {'status': 'error', 'message': 'Analysis already in progress'}, 409
            
        def analysis_thread():
            global analyzer, predictor, is_analyzing, last_update
            try:
                is_analyzing = True
                historical_data = api_manager.get_historical_data()
                analyzer = AdvancedFC25Analyzer(historical_data)
                analyzer.analyze_with_ml()
                predictor = MLBettingPredictor(analyzer)
                last_update = datetime.now()
                is_analyzing = False
            except Exception as e:
                is_analyzing = False
                print(f"Analysis error: {e}")
        
        threading.Thread(target=analysis_thread).start()
        
        return {
            'status': 'success',
            'message': 'Analysis started in background',
            'timestamp': datetime.now().isoformat()
        }

@ns_analysis.route('/status')
class AnalysisStatus(Resource):
    def get(self):
        """Get analysis status"""
        global analyzer, is_analyzing, last_update
        
        status = {
            'is_analyzing': is_analyzing,
            'last_update': last_update.isoformat() if last_update else None,
            'has_analyzer': analyzer is not None,
            'team_count': len(analyzer.team_stats) if analyzer else 0
        }
        
        return {'status': 'success', 'data': status}

@ns_analysis.route('/teams')
class TeamList(Resource):
    def get(self):
        """Get list of all analyzed teams"""
        global analyzer
        
        if not analyzer:
            return {'status': 'error', 'message': 'No analysis data available'}, 404
            
        teams = []
        for team_name, stats in analyzer.team_stats.items():
            teams.append({
                'name': team_name,
                'played': stats['played'],
                'win_rate': stats['win_rate'],
                'elo_rating': analyzer.elo_ratings.get(team_name, 1500),
                'momentum': stats['momentum']
            })
        
        return {
            'status': 'success',
            'teams': sorted(teams, key=lambda x: x['elo_rating'], reverse=True),
            'count': len(teams)
        }

@ns_analysis.route('/team/<string:team_name>')
class TeamAnalysis(Resource):
    def get(self, team_name):
        """Get detailed analysis for a specific team"""
        global analyzer
        
        if not analyzer:
            return {'status': 'error', 'message': 'No analysis data available'}, 404
            
        hist_team = analyzer.find_team_in_data(team_name)
        if not hist_team:
            return {'status': 'error', 'message': f'Team {team_name} not found'}, 404
            
        stats = analyzer.team_stats[hist_team]
        
        return {
            'status': 'success',
            'team': hist_team,
            'stats': {
                'played': stats['played'],
                'won': stats['won'],
                'lost': stats['lost'],
                'draw': stats['draw'],
                'win_rate': stats['win_rate'],
                'avg_goals_for': stats['avg_goals_for'],
                'avg_goals_against': stats['avg_goals_against'],
                'momentum': stats['momentum'],
                'home_record': stats['home_record'],
                'away_record': stats['away_record']
            },
            'elo_rating': analyzer.elo_ratings.get(hist_team, 1500),
            'recent_form': stats['recent_form'][-10:] if stats['recent_form'] else []
        }

@ns_predictions.route('/predict')
class Prediction(Resource):
    @api.expect(prediction_request)
    def post(self):
        """Get prediction for a specific match"""
        global predictor
        
        if not predictor:
            return {'status': 'error', 'message': 'Predictor not initialized. Run analysis first.'}, 404
            
        data = request.get_json()
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        match_data = data.get('match_data')
        
        if not home_team or not away_team:
            return {'status': 'error', 'message': 'home_team and away_team are required'}, 400
        
        try:
            predictions = predictor.predict_match(home_team, away_team, match_data)
            
            if predictions:
                return {
                    'status': 'success',
                    'match': f"{home_team} vs {away_team}",
                    'predictions': predictions,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'success',
                    'match': f"{home_team} vs {away_team}",
                    'predictions': [],
                    'message': 'No confident predictions found',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}, 500

@ns_predictions.route('/upcoming-predictions')
class UpcomingPredictions(Resource):
    def get(self):
        """Get predictions for all upcoming matches"""
        global predictor
        
        if not predictor:
            return {'status': 'error', 'message': 'Predictor not initialized'}, 404
            
        try:
            matches = api_manager.get_upcoming_matches()
            all_predictions = []
            
            for match in matches:
                try:
                    # Extract team names from match data
                    home_team = match.get('opp1')
                    away_team = match.get('opp2')
                    
                    if home_team and away_team:
                        predictions = predictor.predict_match(home_team, away_team, match)
                        
                        if predictions:
                            all_predictions.append({
                                'match': f"{home_team} vs {away_team}",
                                'predictions': predictions,
                                'match_data': {
                                    'time': match.get('time'),
                                    'date': match.get('date')
                                }
                            })
                            
                except Exception as e:
                    print(f"Error predicting match {match.get('opp1', '')} vs {match.get('opp2', '')}: {e}")
                    continue
            
            return {
                'status': 'success',
                'predictions': all_predictions,
                'count': len(all_predictions),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}, 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("🚀 Starting FC25 Prediction API Server...")
    print("📚 API Documentation available at: http://localhost:5000/docs")
    app.run(host='0.0.0.0', port=5000, debug=True)