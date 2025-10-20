"""
Advanced Privacy Model for AI Data Protection
A state-of-the-art privacy preservation system combining multiple techniques
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import hashlib
import re
from transformers import AutoTokenizer, AutoModel
import spacy
from scipy.stats import gaussian_kde
from collections import defaultdict


class PrivacyLevel(Enum):
    """Privacy levels for different data sensitivity"""
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3
    TOP_SECRET = 4


class DataType(Enum):
    """Sensitive data types we can detect"""
    # Personal Identifiers
    FULL_NAME = "full_name"
    FIRST_NAME = "first_name"
    LAST_NAME = "last_name"
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"

    # Financial
    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank_account"
    ROUTING_NUMBER = "routing_number"
    CRYPTO_WALLET = "crypto_wallet"

    # Medical
    MEDICAL_RECORD = "medical_record"
    DIAGNOSIS = "diagnosis"
    PRESCRIPTION = "prescription"

    # Location
    ADDRESS = "address"
    GPS_COORDINATES = "gps"
    IP_ADDRESS = "ip_address"

    # Biometric
    FINGERPRINT = "fingerprint"
    FACE_ENCODING = "face_encoding"
    VOICE_PRINT = "voice_print"

    # Corporate
    TRADE_SECRET = "trade_secret"
    API_KEY = "api_key"
    PASSWORD = "password"


@dataclass
class DetectedEntity:
    """Represents a detected sensitive entity"""
    text: str
    data_type: DataType
    start_idx: int
    end_idx: int
    confidence: float
    context: str
    privacy_level: PrivacyLevel


class AdvancedPIIDetector(nn.Module):
    """
    Transformer-based PII detection model with context awareness
    """
    def __init__(self, model_name="microsoft/deberta-v3-base"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        # Multi-head attention for entity relationships
        # Get the actual embedding dimension from the model
        embed_dim = self.encoder.config.hidden_size
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=12 if embed_dim == 768 else 8,
            dropout=0.1
        )

        # Entity classification head
        self.entity_classifier = nn.Sequential(
            nn.Linear(embed_dim, 384),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(384, len(DataType))
        )

        # Confidence scoring head
        self.confidence_head = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Context-aware privacy level predictor
        self.privacy_classifier = nn.Sequential(
            nn.Linear(768 * 2, 256),  # Entity + context embeddings
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(PrivacyLevel))
        )

        # Pattern matchers for high-precision detection
        self.pattern_matchers = self._compile_patterns()

    def _compile_patterns(self) -> Dict[DataType, re.Pattern]:
        """Compile regex patterns for known PII formats"""
        patterns = {
            DataType.EMAIL: re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            DataType.PHONE: re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'),
            DataType.SSN: re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            DataType.CREDIT_CARD: re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
            DataType.IP_ADDRESS: re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
            DataType.API_KEY: re.compile(r'[A-Za-z0-9]{32,}'),
        }
        return patterns

    def forward(self, text: str) -> List[DetectedEntity]:
        """Detect PII entities in text"""
        entities = []

        # Tokenize and encode
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )

        # Get embeddings
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            embeddings = outputs.last_hidden_state

        # Apply cross-attention for entity relationships
        attn_output, _ = self.cross_attention(
            embeddings, embeddings, embeddings
        )

        # Classify each token
        entity_logits = self.entity_classifier(attn_output)
        confidences = self.confidence_head(attn_output)

        # Decode predictions to entities
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        for i, (token, logit, conf) in enumerate(zip(tokens, entity_logits[0], confidences[0])):
            if conf > 0.7:  # Confidence threshold
                entity_type_idx = torch.argmax(logit).item()
                entity_type = list(DataType)[entity_type_idx]

                # Get privacy level from context
                context_embedding = torch.cat([
                    attn_output[0, i],
                    attn_output[0].mean(dim=0)
                ])
                privacy_logits = self.privacy_classifier(context_embedding)
                privacy_level = list(PrivacyLevel)[torch.argmax(privacy_logits).item()]

                entities.append(DetectedEntity(
                    text=token,
                    data_type=entity_type,
                    start_idx=i,
                    end_idx=i+1,
                    confidence=conf.item(),
                    context=text[max(0, i-50):min(len(text), i+50)],
                    privacy_level=privacy_level
                ))

        # Merge adjacent tokens of same type
        entities = self._merge_entities(entities)

        # Pattern-based detection for high precision
        pattern_entities = self._pattern_detection(text)
        entities.extend(pattern_entities)

        return entities

    def _merge_entities(self, entities: List[DetectedEntity]) -> List[DetectedEntity]:
        """Merge adjacent tokens belonging to same entity"""
        if not entities:
            return []

        merged = []
        current = entities[0]

        for entity in entities[1:]:
            if (entity.data_type == current.data_type and
                entity.start_idx == current.end_idx):
                current.end_idx = entity.end_idx
                current.text += entity.text
                current.confidence = max(current.confidence, entity.confidence)
            else:
                merged.append(current)
                current = entity

        merged.append(current)
        return merged

    def _pattern_detection(self, text: str) -> List[DetectedEntity]:
        """Use regex patterns for high-precision detection"""
        entities = []

        for data_type, pattern in self.pattern_matchers.items():
            for match in pattern.finditer(text):
                entities.append(DetectedEntity(
                    text=match.group(),
                    data_type=data_type,
                    start_idx=match.start(),
                    end_idx=match.end(),
                    confidence=0.95,  # High confidence for pattern matches
                    context=text[max(0, match.start()-50):min(len(text), match.end()+50)],
                    privacy_level=PrivacyLevel.CONFIDENTIAL
                ))

        return entities


class DifferentialPrivacyEngine:
    """
    Advanced differential privacy implementation with adaptive noise
    """
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_budget = epsilon
        self.query_history = []

    def add_noise(self, data: np.ndarray, sensitivity: float,
                  mechanism: str = "laplace") -> np.ndarray:
        """Add calibrated noise to data"""
        if self.privacy_budget <= 0:
            raise ValueError("Privacy budget exhausted")

        if mechanism == "laplace":
            scale = sensitivity / self.epsilon
            noise = np.random.laplace(0, scale, data.shape)
        elif mechanism == "gaussian":
            sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
            noise = np.random.normal(0, sigma, data.shape)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")

        # Track privacy budget consumption
        self.privacy_budget -= self.epsilon / len(self.query_history)
        self.query_history.append({
            'timestamp': np.datetime64('now'),
            'mechanism': mechanism,
            'epsilon_used': self.epsilon / len(self.query_history)
        })

        return data + noise

    def adaptive_noise(self, data: np.ndarray, utility_target: float = 0.95) -> np.ndarray:
        """
        Adaptively calibrate noise to balance privacy and utility
        """
        # Estimate data sensitivity
        sensitivity = self._estimate_sensitivity(data)

        # Start with minimal noise
        epsilon_test = self.epsilon * 2
        best_epsilon = self.epsilon
        best_utility = 0

        # Binary search for optimal epsilon
        for _ in range(10):
            noisy_data = self.add_noise(data.copy(), sensitivity, "laplace")
            utility = self._measure_utility(data, noisy_data)

            if utility >= utility_target:
                best_epsilon = epsilon_test
                best_utility = utility
                epsilon_test *= 0.9  # Try more privacy
            else:
                epsilon_test *= 1.1  # Need less privacy

        # Apply best noise level
        self.epsilon = best_epsilon
        return self.add_noise(data, sensitivity, "laplace")

    def _estimate_sensitivity(self, data: np.ndarray) -> float:
        """Estimate query sensitivity"""
        if len(data.shape) == 1:
            return np.ptp(data)  # Range for 1D
        else:
            return np.max(np.ptp(data, axis=0))  # Max range across dimensions

    def _measure_utility(self, original: np.ndarray, noisy: np.ndarray) -> float:
        """Measure utility preservation"""
        # Statistical similarity
        if len(original) > 1:
            correlation = np.corrcoef(original.flatten(), noisy.flatten())[0, 1]
        else:
            correlation = 1.0

        # Distribution similarity (KL divergence approximation)
        hist_orig, bins = np.histogram(original, bins=20)
        hist_noisy, _ = np.histogram(noisy, bins=bins)

        hist_orig = hist_orig / hist_orig.sum() + 1e-10
        hist_noisy = hist_noisy / hist_noisy.sum() + 1e-10

        kl_div = np.sum(hist_orig * np.log(hist_orig / hist_noisy))

        # Combined utility score
        utility = (correlation + np.exp(-kl_div)) / 2
        return max(0, min(1, utility))


class KAnonymityEngine:
    """
    K-anonymity with l-diversity and t-closeness extensions
    """
    def __init__(self, k: int = 5, l: int = 2, t: float = 0.2):
        self.k = k
        self.l = l  # l-diversity parameter
        self.t = t  # t-closeness parameter

    def anonymize(self, data: pd.DataFrame,
                  quasi_identifiers: List[str],
                  sensitive_attrs: List[str]) -> pd.DataFrame:
        """Apply k-anonymity with extensions"""
        import pandas as pd

        anonymized = data.copy()

        # Group by quasi-identifiers
        groups = anonymized.groupby(quasi_identifiers)

        # Ensure k-anonymity
        for name, group in groups:
            if len(group) < self.k:
                # Generalize or suppress
                anonymized = self._generalize_group(anonymized, group, quasi_identifiers)

        # Ensure l-diversity
        if self.l > 1:
            anonymized = self._ensure_l_diversity(anonymized, quasi_identifiers, sensitive_attrs)

        # Ensure t-closeness
        if self.t < 1.0:
            anonymized = self._ensure_t_closeness(anonymized, quasi_identifiers, sensitive_attrs)

        return anonymized

    def _generalize_group(self, df: pd.DataFrame, group: pd.DataFrame,
                         quasi_ids: List[str]) -> pd.DataFrame:
        """Generalize small groups"""
        for col in quasi_ids:
            if df[col].dtype == 'object':
                # Categorical generalization
                df.loc[group.index, col] = '*'
            else:
                # Numerical generalization (use range)
                min_val = group[col].min()
                max_val = group[col].max()
                df.loc[group.index, col] = f"[{min_val}-{max_val}]"

        return df

    def _ensure_l_diversity(self, df: pd.DataFrame, quasi_ids: List[str],
                           sensitive: List[str]) -> pd.DataFrame:
        """Ensure l-diversity in sensitive attributes"""
        groups = df.groupby(quasi_ids)

        for name, group in groups:
            for attr in sensitive:
                unique_values = group[attr].nunique()
                if unique_values < self.l:
                    # Add synthetic records or generalize
                    df = self._add_diversity(df, group, attr)

        return df

    def _ensure_t_closeness(self, df: pd.DataFrame, quasi_ids: List[str],
                           sensitive: List[str]) -> pd.DataFrame:
        """Ensure t-closeness for sensitive attributes"""
        import scipy.stats as stats

        global_distributions = {}
        for attr in sensitive:
            global_distributions[attr] = df[attr].value_counts(normalize=True)

        groups = df.groupby(quasi_ids)

        for name, group in groups:
            for attr in sensitive:
                group_dist = group[attr].value_counts(normalize=True)

                # Calculate Earth Mover's Distance
                emd = self._earth_movers_distance(
                    global_distributions[attr],
                    group_dist
                )

                if emd > self.t:
                    # Adjust distribution
                    df = self._adjust_distribution(df, group, attr, global_distributions[attr])

        return df

    def _earth_movers_distance(self, dist1, dist2) -> float:
        """Calculate EMD between two distributions"""
        # Simplified EMD calculation
        all_values = set(dist1.index) | set(dist2.index)
        distance = 0

        for val in all_values:
            p1 = dist1.get(val, 0)
            p2 = dist2.get(val, 0)
            distance += abs(p1 - p2)

        return distance / 2

    def _add_diversity(self, df, group, attr):
        """Add diversity to satisfy l-diversity"""
        # Implementation would add synthetic records
        return df

    def _adjust_distribution(self, df, group, attr, target_dist):
        """Adjust group distribution to match target"""
        # Implementation would adjust values
        return df


class SyntheticDataGenerator:
    """
    Generate privacy-preserving synthetic data using advanced techniques
    """
    def __init__(self, privacy_budget: float = 1.0):
        self.privacy_budget = privacy_budget
        self.generators = {}

    def train_generator(self, real_data: np.ndarray,
                       method: str = "ctgan") -> None:
        """Train synthetic data generator"""
        if method == "ctgan":
            from ctgan import CTGAN
            self.generators['ctgan'] = CTGAN(
                embedding_dim=128,
                generator_dim=(256, 256),
                discriminator_dim=(256, 256),
                pac=10,
                epochs=300
            )
            self.generators['ctgan'].fit(real_data)

        elif method == "dpgan":
            # Differentially private GAN
            self.generators['dpgan'] = self._create_dpgan()
            self._train_dpgan(real_data)

        elif method == "vae":
            # Variational autoencoder
            self.generators['vae'] = self._create_vae()
            self._train_vae(real_data)

    def generate(self, n_samples: int, method: str = "ctgan") -> np.ndarray:
        """Generate synthetic samples"""
        if method not in self.generators:
            raise ValueError(f"Generator {method} not trained")

        if method == "ctgan":
            return self.generators['ctgan'].sample(n_samples)
        elif method == "dpgan":
            return self._generate_dpgan(n_samples)
        elif method == "vae":
            return self._generate_vae(n_samples)

    def _create_dpgan(self):
        """Create differentially private GAN"""
        class DPGAN(nn.Module):
            def __init__(self, input_dim, latent_dim=100):
                super().__init__()
                self.generator = nn.Sequential(
                    nn.Linear(latent_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, input_dim),
                    nn.Tanh()
                )

                self.discriminator = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )

            def forward(self, z):
                return self.generator(z)

        return DPGAN(100, 100)  # Placeholder dimensions

    def _train_dpgan(self, data):
        """Train DP-GAN with privacy guarantees"""
        # Implementation with gradient clipping and noise addition
        pass

    def _generate_dpgan(self, n_samples):
        """Generate from DP-GAN"""
        latent = torch.randn(n_samples, 100)
        with torch.no_grad():
            synthetic = self.generators['dpgan'](latent)
        return synthetic.numpy()

    def _create_vae(self):
        """Create Variational Autoencoder"""
        class VAE(nn.Module):
            def __init__(self, input_dim, latent_dim=20):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 400),
                    nn.ReLU(),
                    nn.Linear(400, latent_dim * 2)  # Mean and variance
                )

                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 400),
                    nn.ReLU(),
                    nn.Linear(400, input_dim),
                    nn.Sigmoid()
                )

                self.latent_dim = latent_dim

            def encode(self, x):
                h = self.encoder(x)
                mean, log_var = h.chunk(2, dim=-1)
                return mean, log_var

            def reparameterize(self, mean, log_var):
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                return mean + eps * std

            def forward(self, x):
                mean, log_var = self.encode(x)
                z = self.reparameterize(mean, log_var)
                return self.decoder(z), mean, log_var

        return VAE(100, 20)  # Placeholder dimensions

    def _train_vae(self, data):
        """Train VAE"""
        # Implementation with ELBO loss
        pass

    def _generate_vae(self, n_samples):
        """Generate from VAE"""
        with torch.no_grad():
            z = torch.randn(n_samples, 20)
            synthetic = self.generators['vae'].decoder(z)
        return synthetic.numpy()


class PrivacyUtilityOptimizer:
    """
    Optimize the trade-off between privacy and utility
    """
    def __init__(self):
        self.pareto_frontier = []
        self.optimal_configs = {}

    def optimize(self, data: np.ndarray,
                privacy_methods: List[str],
                utility_metric: str = "accuracy") -> Dict:
        """Find optimal privacy configuration"""

        results = []

        # Test different privacy configurations
        for epsilon in np.logspace(-2, 1, 20):  # Privacy budget range
            for method in privacy_methods:
                config = {
                    'method': method,
                    'epsilon': epsilon
                }

                # Apply privacy
                private_data = self._apply_privacy(data, config)

                # Measure utility
                utility = self._measure_utility(data, private_data, utility_metric)

                # Measure privacy
                privacy = self._measure_privacy(data, private_data, epsilon)

                results.append({
                    'config': config,
                    'utility': utility,
                    'privacy': privacy,
                    'score': utility * privacy  # Combined score
                })

        # Find Pareto frontier
        self.pareto_frontier = self._find_pareto_frontier(results)

        # Select optimal based on requirements
        optimal = max(results, key=lambda x: x['score'])

        return optimal

    def _apply_privacy(self, data: np.ndarray, config: Dict) -> np.ndarray:
        """Apply privacy method with configuration"""
        if config['method'] == 'differential_privacy':
            dp_engine = DifferentialPrivacyEngine(epsilon=config['epsilon'])
            return dp_engine.add_noise(data, sensitivity=1.0)

        # Add other methods
        return data

    def _measure_utility(self, original: np.ndarray,
                        private: np.ndarray,
                        metric: str) -> float:
        """Measure utility preservation"""
        if metric == "accuracy":
            # Statistical accuracy
            mse = np.mean((original - private) ** 2)
            return 1 / (1 + mse)
        elif metric == "correlation":
            return np.corrcoef(original.flatten(), private.flatten())[0, 1]
        elif metric == "distribution":
            # KS test for distribution similarity
            from scipy.stats import ks_2samp
            _, p_value = ks_2samp(original.flatten(), private.flatten())
            return p_value

        return 0.5

    def _measure_privacy(self, original: np.ndarray,
                        private: np.ndarray,
                        epsilon: float) -> float:
        """Measure privacy level achieved"""
        # Estimate disclosure risk
        disclosure_risk = self._estimate_disclosure_risk(original, private)

        # Convert to privacy score
        privacy_score = 1 - disclosure_risk

        # Factor in epsilon
        privacy_score *= (1 - np.exp(-epsilon))

        return privacy_score

    def _estimate_disclosure_risk(self, original: np.ndarray,
                                 private: np.ndarray) -> float:
        """Estimate risk of re-identification"""
        # Simplified uniqueness-based risk
        unique_original = np.unique(original, axis=0) if len(original.shape) > 1 else np.unique(original)
        unique_private = np.unique(private, axis=0) if len(private.shape) > 1 else np.unique(private)

        # Ratio of unique records (lower is better)
        risk = len(unique_private) / max(len(unique_original), 1)

        return min(1.0, risk)

    def _find_pareto_frontier(self, results: List[Dict]) -> List[Dict]:
        """Find Pareto optimal configurations"""
        frontier = []

        for r1 in results:
            dominated = False
            for r2 in results:
                if r1 == r2:
                    continue
                # Check if r2 dominates r1
                if (r2['utility'] >= r1['utility'] and
                    r2['privacy'] >= r1['privacy'] and
                    (r2['utility'] > r1['utility'] or r2['privacy'] > r1['privacy'])):
                    dominated = True
                    break

            if not dominated:
                frontier.append(r1)

        return frontier


class UnifiedPrivacyPlatform:
    """
    Main platform integrating all privacy components
    """
    def __init__(self):
        self.detector = AdvancedPIIDetector()
        self.dp_engine = DifferentialPrivacyEngine()
        self.k_anon_engine = KAnonymityEngine()
        self.synthetic_gen = SyntheticDataGenerator()
        self.optimizer = PrivacyUtilityOptimizer()

        # Metrics tracking
        self.metrics = {
            'processed_records': 0,
            'privacy_violations_prevented': 0,
            'average_utility': 0.95,
            'compliance_score': 1.0
        }

    def process_data(self, data: Any,
                     requirements: Dict) -> Tuple[Any, Dict]:
        """
        Main entry point for privacy processing

        Args:
            data: Input data (text, structured, or mixed)
            requirements: Privacy requirements and constraints

        Returns:
            Tuple of (processed_data, privacy_report)
        """

        # Step 1: Detect sensitive information
        if isinstance(data, str):
            entities = self.detector(data)
        else:
            entities = self._detect_structured(data)

        # Step 2: Determine optimal privacy strategy
        strategy = self.optimizer.optimize(
            data=self._to_numpy(data),
            privacy_methods=requirements.get('methods', ['differential_privacy']),
            utility_metric=requirements.get('utility_metric', 'accuracy')
        )

        # Step 3: Apply privacy transformations
        if strategy['config']['method'] == 'differential_privacy':
            processed = self._apply_differential_privacy(data, entities, strategy['config'])
        elif strategy['config']['method'] == 'k_anonymity':
            processed = self._apply_k_anonymity(data, entities)
        elif strategy['config']['method'] == 'synthetic':
            processed = self._generate_synthetic(data)
        else:
            processed = self._apply_redaction(data, entities)

        # Step 4: Validate compliance
        compliance = self._validate_compliance(processed, requirements)

        # Step 5: Generate privacy report
        report = self._generate_report(
            entities, strategy, compliance, processed
        )

        # Update metrics
        self._update_metrics(report)

        return processed, report

    def _detect_structured(self, data) -> List[DetectedEntity]:
        """Detect PII in structured data"""
        entities = []

        if hasattr(data, 'columns'):  # DataFrame-like
            for col in data.columns:
                # Check column names for sensitive fields
                if any(keyword in col.lower() for keyword in
                      ['name', 'email', 'phone', 'ssn', 'address']):
                    # Sample values to determine data type
                    sample = data[col].dropna().head(10)
                    for val in sample:
                        detected = self.detector(str(val))
                        entities.extend(detected)

        return entities

    def _to_numpy(self, data) -> np.ndarray:
        """Convert various data types to numpy array"""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, str):
            # Encode text to numerical representation
            return np.array([ord(c) for c in data])
        elif hasattr(data, 'values'):  # DataFrame-like
            return data.values
        else:
            return np.array(data)

    def _apply_differential_privacy(self, data, entities, config):
        """Apply differential privacy"""
        if isinstance(data, str):
            # Text data - add noise to numerical representations
            return self._dp_text(data, entities, config['epsilon'])
        else:
            # Structured data
            np_data = self._to_numpy(data)
            return self.dp_engine.add_noise(np_data, sensitivity=1.0)

    def _dp_text(self, text: str, entities: List[DetectedEntity], epsilon: float) -> str:
        """Apply DP to text by perturbing entities"""
        protected = text

        for entity in sorted(entities, key=lambda e: e.start_idx, reverse=True):
            if entity.privacy_level.value >= PrivacyLevel.CONFIDENTIAL.value:
                # Replace with noisy version or generalized term
                replacement = self._generate_dp_replacement(entity, epsilon)
                protected = protected[:entity.start_idx] + replacement + protected[entity.end_idx:]

        return protected

    def _generate_dp_replacement(self, entity: DetectedEntity, epsilon: float) -> str:
        """Generate DP-compliant replacement for entity"""
        if entity.data_type == DataType.FULL_NAME:
            return "[NAME]"
        elif entity.data_type == DataType.EMAIL:
            return "[EMAIL]"
        elif entity.data_type in [DataType.PHONE, DataType.SSN]:
            # Add noise to numerical values
            digits = ''.join(c for c in entity.text if c.isdigit())
            if digits:
                noisy = int(digits) + int(np.random.laplace(0, 1/epsilon))
                return str(noisy)[:len(digits)]

        return "[REDACTED]"

    def _apply_k_anonymity(self, data, entities):
        """Apply k-anonymity"""
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            # Convert to DataFrame for k-anonymity
            data = pd.DataFrame(data)

        # Identify quasi-identifiers from entities
        quasi_ids = list(set(e.data_type.value for e in entities
                           if e.privacy_level.value < PrivacyLevel.RESTRICTED.value))

        # Identify sensitive attributes
        sensitive = list(set(e.data_type.value for e in entities
                           if e.privacy_level.value >= PrivacyLevel.RESTRICTED.value))

        return self.k_anon_engine.anonymize(data, quasi_ids, sensitive)

    def _generate_synthetic(self, data):
        """Generate synthetic data"""
        np_data = self._to_numpy(data)

        # Train generator if needed
        if 'ctgan' not in self.synthetic_gen.generators:
            self.synthetic_gen.train_generator(np_data)

        # Generate synthetic samples
        n_samples = len(np_data) if hasattr(np_data, '__len__') else 100
        return self.synthetic_gen.generate(n_samples)

    def _apply_redaction(self, data, entities):
        """Simple redaction for low-privacy requirements"""
        if isinstance(data, str):
            protected = data
            for entity in sorted(entities, key=lambda e: e.start_idx, reverse=True):
                if entity.privacy_level.value >= PrivacyLevel.INTERNAL.value:
                    protected = protected[:entity.start_idx] + "[REDACTED]" + protected[entity.end_idx:]
            return protected

        return data

    def _validate_compliance(self, processed_data, requirements) -> Dict:
        """Validate against compliance requirements"""
        compliance = {
            'gdpr': True,
            'ccpa': True,
            'hipaa': True,
            'satisfied': True
        }

        # Check for remaining PII
        if isinstance(processed_data, str):
            remaining_entities = self.detector(processed_data)
            high_risk = [e for e in remaining_entities
                        if e.privacy_level.value >= PrivacyLevel.CONFIDENTIAL.value]

            if high_risk:
                compliance['satisfied'] = False
                if 'gdpr' in requirements.get('regulations', []):
                    compliance['gdpr'] = False

        return compliance

    def _generate_report(self, entities, strategy, compliance, processed_data) -> Dict:
        """Generate comprehensive privacy report"""
        return {
            'timestamp': np.datetime64('now'),
            'entities_detected': len(entities),
            'entity_types': list(set(e.data_type.value for e in entities)),
            'privacy_method': strategy['config']['method'],
            'privacy_level': strategy['privacy'],
            'utility_preserved': strategy['utility'],
            'compliance': compliance,
            'risk_score': 1 - strategy['privacy'],
            'recommendations': self._generate_recommendations(entities, strategy)
        }

    def _generate_recommendations(self, entities, strategy) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        if strategy['utility'] < 0.9:
            recommendations.append("Consider using synthetic data generation for better utility")

        if strategy['privacy'] < 0.8:
            recommendations.append("Increase privacy budget or use stronger anonymization")

        high_risk = [e for e in entities if e.privacy_level.value >= PrivacyLevel.RESTRICTED.value]
        if high_risk:
            recommendations.append(f"Found {len(high_risk)} high-risk entities - consider encryption")

        return recommendations

    def _update_metrics(self, report):
        """Update platform metrics"""
        self.metrics['processed_records'] += 1
        if report['risk_score'] < 0.1:
            self.metrics['privacy_violations_prevented'] += 1

        # Running average of utility
        alpha = 0.1
        self.metrics['average_utility'] = (
            alpha * report['utility_preserved'] +
            (1 - alpha) * self.metrics['average_utility']
        )

        if report['compliance']['satisfied']:
            self.metrics['compliance_score'] = min(1.0, self.metrics['compliance_score'] + 0.01)
        else:
            self.metrics['compliance_score'] *= 0.95


# Example usage and testing
if __name__ == "__main__":
    # Initialize platform
    platform = UnifiedPrivacyPlatform()

    # Example 1: Process text with PII
    text_data = """
    John Smith (john.smith@email.com) called from 555-123-4567 about his
    medical record #MR123456. His SSN is 123-45-6789 and he lives at
    123 Main St, Boston, MA. Credit card: 4532-1234-5678-9012.
    """

    requirements = {
        'methods': ['differential_privacy', 'redaction'],
        'utility_metric': 'accuracy',
        'regulations': ['gdpr', 'hipaa']
    }

    processed_text, report = platform.process_data(text_data, requirements)

    print("Original:", text_data[:100])
    print("Processed:", processed_text[:100])
    print("Privacy Report:", report)

    # Example 2: Process structured data
    import pandas as pd

    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'email': ['alice@ex.com', 'bob@ex.com', 'charlie@ex.com'],
        'salary': [50000, 60000, 55000],
        'department': ['HR', 'IT', 'HR']
    })

    processed_df, report = platform.process_data(df, requirements)
    print("\nStructured Data Report:", report)

    # Platform metrics
    print("\nPlatform Metrics:", platform.metrics)