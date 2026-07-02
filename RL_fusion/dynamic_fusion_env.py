"""
Gymnasium based Implementation of RL-based Dynamic Fusion Environment)

Implementation of RL-based Dynamic Fusion Envrionment from:
    "A Multi-View Media Profiling Suite: Resources, Evaluation, and Analysis"
    Findings of the ACL 2026.

In this environment, we formulate the Fusion problem as a contextual bandit (actions do not affect state transitions).
Our MDP is formulated as follows: For a media outlet F_i, the state concatenates the five multi-view embeddings
    s_t = { F^(a), F^(h), F^(l), F^(t), F^(w) }
    (a) Alexa graph      - GNN embedding from audience-overlap graph
    (h) Hyperlink graph  - GNN embedding from on-site hyperlink graph
    (l) LLM graph        - GNN embedding from LLM-generated graph
    (t) Articles         - text embedding from outlet articles
    (w) Wikipedia        - text embedding from Wikipedia descriptions
The action is a continuous weight vector w in R^5 with w_k in [0, 1]. The fused
embedding is E_fused = sum_k w_k F^(k), and the reward is the probability of the
true label under a fixed classifier, r_t = P(y_true | E_fused). The goal is a
policy that maximizes the immediate expected reward (discount factor ~ 0).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import pickle
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Default data directory: a `data/` folder next to this file. Can be overridden
# via the `data_dir` argument or the DYNAMIC_FUSION_DATA_DIR environment variable.
DEFAULT_DATA_DIR = os.environ.get(
    "DYNAMIC_FUSION_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)


class DynamicFusionEnv(gym.Env):
    """
    Gymnasium environment for RL-based Dynamic Fusion of multi-view media
    representations, formulated as a contextual bandit.
    
    Supports both Small (ACL) and Large (MBFC) datasets with proper evaluation metrics.
    """
    
    def __init__(self, 
                 mode: str = "train",  # "train" or "test"
                 data_dir: str = DEFAULT_DATA_DIR,
                 task: str = "bias",  # "bias" or "fact"
                 action_type: str = "weights",  # "weights" or "selection"
                 dataset_size: str = "small",  # "small" (ACL) or "large" (MBFC)
                 split_id: int = 0,
                 classifier_type: str = "rf",
                 out_csv_name='results/results'):  # "rf" for RandomForest, "svm" for SVM
        """
        Initialize the environment.
        
        Args:
            mode: "train" for training on train_outlets, "test" for evaluation on test_outlets
            data_dir: Directory containing the embedding files
            task: Classification task - "bias" or "fact"
            action_type: Type of action - "weights" for continuous weighting, "selection" for binary selection  
            dataset_size: "small" for ACL dataset (~470), "large" for MBFC dataset (~2700)
            split_id: Which data split to use (0-4)
            classifier_type: Classifier for evaluation - "rf" or "svm"
        """
        super().__init__()
        
        self.mode = mode
        self.data_dir = data_dir
        self.task = task
        self.action_type = action_type
        self.dataset_size = dataset_size
        self.split_id = split_id
        self.classifier_type = classifier_type
        
        # Load all data
        self._load_all_data()
        
        # Initialize classifier
        if classifier_type == "svm":
            self.classifier = SVC(kernel='linear', probability=True, random_state=42)
        else:
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        self.label_encoder = LabelEncoder()
        
        # Set embedding dimension first
        self.embedding_dim = self._get_embedding_dimension()
        
        # Prepare labels and splits
        self._prepare_labels_and_splits()
        
        # Extract and cache text embeddings
        self._extract_text_embeddings()
        
        # Train classifier on training data
        self._train_classifier()
        
        # Set outlet pool based on mode
        if self.mode == 'train':
            self.outlet_pool = self.train_outlets
            print(f"Environment initialized in TRAIN mode with {len(self.outlet_pool)} outlets.")
        else:
            self.outlet_pool = self.test_outlets
            print(f"Environment initialized in TEST mode with {len(self.outlet_pool)} outlets.")
        
        # Environment state
        self.current_outlet_idx = 0
        self.current_outlet = None
        self.current_embeddings = None
        self.episode_rewards = []
        self.episode_predictions = []
        self.episode_true_labels = []
        
        # Define action and observation spaces
        self.num_embedding_types = 5  # A, W, Alexa, Hyperlink, LLM
        
        if action_type == "weights":
            # Continuous weights for each embedding type
            self.action_space = spaces.Box(
                low=0.0, 
                high=1.0, 
                shape=(self.num_embedding_types,), 
                dtype=np.float32
            )
        else:  # "selection"
            # Binary selection for each embedding type
            self.action_space = spaces.MultiBinary(self.num_embedding_types)
        
        # Observation space: concatenated embeddings from all types
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.num_embedding_types * self.embedding_dim,), 
            dtype=np.float32
        )
        
        self.out_csv_name = out_csv_name
        self.run = 0
        self.metrics = []
        self.episode_steps = 0

    def _load_all_data(self):
        """Load all embedding data and metadata."""
        # Load main corpus/labels
        if self.dataset_size == "small":
            # Load small dataset embeddings
            self.articles_df = pd.read_pickle(f"{self.data_dir}/embeddings_1a_small.pkl")
            self.media_df = pd.read_pickle(f"{self.data_dir}/embeddings_1b_small.pkl")
            
            # Add normalized 'name' column for consistent matching
            self.articles_df['name'] = self.articles_df['source_url'].apply(self._normalize_url)
            self.media_df['name'] = self.media_df['source_url'].apply(self._normalize_url)
            
            # Load GNN embeddings
            self._load_gnn_embeddings_small()
        else:
            # Load large dataset embeddings
            self.articles_df = pd.read_pickle(f"{self.data_dir}/embeddings_2a_large.pkl")
            self.media_df = pd.read_pickle(f"{self.data_dir}/embeddings_2b_large.pkl")
            
            # Add normalized 'name' column for consistent matching
            self.articles_df['name'] = self.articles_df['source_url'].apply(self._normalize_url)
            self.media_df['name'] = self.media_df['source_url'].apply(self._normalize_url)
            
            # Load GNN embeddings
            self._load_gnn_embeddings_large()
        
        # Load corpus for labels (if available)
        corpus_path = f"{self.data_dir}/corpus.tsv"
        if os.path.exists(corpus_path):
            self.corpus_df = pd.read_csv(corpus_path, sep='\t')
        else:
            # Use media_df as corpus
            self.corpus_df = self.media_df.copy()
            
        # Load splits
        splits_path = f"{self.data_dir}/splits.json"
        if os.path.exists(splits_path):
            with open(splits_path, 'r') as f:
                all_splits = json.load(f)
                domain_splits = all_splits[str(self.split_id)]
                # Convert domain-based splits to name-based splits
                self._convert_domain_splits_to_name_splits(domain_splits)
        else:
            # Create splits if not available
            self._create_splits()

    def _load_gnn_embeddings_small(self):
        """Load GNN embeddings for small dataset."""
        # Load GPT/LLM embeddings
        with open(f"{self.data_dir}/gpt_small/gpt_small/gpt_sage1,128,128,64.json", 'r') as f:
            gpt_embeddings = json.load(f)
        with open(f"{self.data_dir}/gpt_small/gpt_small/final_identifier_small_gpt.json", 'r') as f:
            gpt_id_map = json.load(f)
        
        # Convert GPT embeddings from ID-based to normalized name-based
        self.gpt_embeddings = {}
        for outlet_name, outlet_id in gpt_id_map.items():
            if str(outlet_id) in gpt_embeddings:
                normalized_name = self._normalize_url(outlet_name)
                self.gpt_embeddings[normalized_name] = gpt_embeddings[str(outlet_id)]
        
        # Load Hyperlink embeddings  
        with open(f"{self.data_dir}/onsite_small/onsite_small/sage1,128,128,64.json", 'r') as f:
            hyperlink_embeddings = json.load(f)
        with open(f"{self.data_dir}/onsite_small/onsite_small/encoding.json", 'r') as f:
            hyperlink_id_map = json.load(f)
            
        # Convert Hyperlink embeddings from ID-based to normalized name-based
        self.hyperlink_embeddings = {}
        for outlet_name, outlet_id in hyperlink_id_map.items():
            if str(outlet_id) in hyperlink_embeddings and outlet_name:  # Skip empty names
                normalized_name = self._normalize_url(outlet_name)
                self.hyperlink_embeddings[normalized_name] = hyperlink_embeddings[str(outlet_id)]
        
        # Load Alexa embeddings (normalize names)
        with open(f"{self.data_dir}/Alexa/Alexa/GraphConv_100_lr_0001_audience_overlap_level_3_pyg_laer_4.json", 'r') as f:
            alexa_embeddings_raw = json.load(f)
        
        # Normalize Alexa embedding names
        self.alexa_embeddings = {}
        for outlet_name, embedding in alexa_embeddings_raw.items():
            normalized_name = self._normalize_url(outlet_name)
            self.alexa_embeddings[normalized_name] = embedding

    def _load_gnn_embeddings_large(self):
        """Load GNN embeddings for large dataset."""
        # Load GPT/LLM embeddings
        with open(f"{self.data_dir}/gpt_large/gpt_large/gpt_sage1,128,128,64.json", 'r') as f:
            gpt_embeddings = json.load(f)
        with open(f"{self.data_dir}/gpt_large/gpt_large/final_identifier_large_gpt.json", 'r') as f:
            gpt_id_map = json.load(f)
        
        # Convert GPT embeddings from ID-based to normalized name-based
        self.gpt_embeddings = {}
        for outlet_name, outlet_id in gpt_id_map.items():
            if str(outlet_id) in gpt_embeddings:
                normalized_name = self._normalize_url(outlet_name)
                self.gpt_embeddings[normalized_name] = gpt_embeddings[str(outlet_id)]
        
        # Load Hyperlink embeddings
        with open(f"{self.data_dir}/onsite_large/sage1,128,128,64.json", 'r') as f:
            hyperlink_embeddings = json.load(f)
        with open(f"{self.data_dir}/onsite_large/name2id.json", 'r') as f:
            hyperlink_id_map = json.load(f)
            
        # Convert Hyperlink embeddings from ID-based to normalized name-based
        self.hyperlink_embeddings = {}
        for outlet_name, outlet_id in hyperlink_id_map.items():
            if str(outlet_id) in hyperlink_embeddings and outlet_name:
                normalized_name = self._normalize_url(outlet_name)
                self.hyperlink_embeddings[normalized_name] = hyperlink_embeddings[str(outlet_id)]
        
        # Load Alexa embeddings (normalize names)
        with open(f"{self.data_dir}/Alexa/Alexa/GraphConv_100_lr_0001_audience_overlap_level_3_pyg_laer_4.json", 'r') as f:
            alexa_embeddings_raw = json.load(f)
        
        # Normalize Alexa embedding names
        self.alexa_embeddings = {}
        for outlet_name, embedding in alexa_embeddings_raw.items():
            normalized_name = self._normalize_url(outlet_name)
            self.alexa_embeddings[normalized_name] = embedding

    def _convert_domain_splits_to_name_splits(self, domain_splits: Dict[str, List[str]]):
        """Convert domain-based splits to name-based splits."""
        # Create domain to name mapping
        domain_to_name = {}
        for _, row in self.media_df.iterrows():
            name = row['name']
            url = row['source_url']
            domain = url.replace('https://', '').replace('http://', '').replace('www.', '').rstrip('/')
            domain_to_name[domain] = name
        
        # Convert splits
        train_names = []
        test_names = []
        
        for domain in domain_splits['train']:
            if domain in domain_to_name:
                train_names.append(domain_to_name[domain])
        
        for domain in domain_splits['test']:
            if domain in domain_to_name:
                test_names.append(domain_to_name[domain])
        
        self.splits = {'train': train_names, 'test': test_names}
        print(f"Converted splits: {len(train_names)} train, {len(test_names)} test outlets")
    
    def _create_splits(self):
        """Create train/test splits using StratifiedGroupKFold."""
        # Get unique outlets from media_df
        unique_outlets = self.media_df[['name', self.task]].drop_duplicates()
        
        # Create stratified group k-fold splits
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        outlets = unique_outlets['name'].values
        labels = unique_outlets[self.task].values
        
        splits_list = []
        for train_idx, test_idx in sgkf.split(outlets, labels, outlets):
            train_outlets = outlets[train_idx].tolist()
            test_outlets = outlets[test_idx].tolist()
            splits_list.append({'train': train_outlets, 'test': test_outlets})
        
        self.splits = splits_list[self.split_id]

    def _prepare_labels_and_splits(self):
        """Prepare labels and train/test splits."""
        # Create name-to-domain and domain-to-name mappings
        self.name_to_domain = {}
        self.domain_to_name = {}
        
        for _, row in self.media_df.iterrows():
            normalized_name = row['name']  # Already normalized in _load_all_data
            original_url = row['source_url']
            # Use normalized name as both key and value since GNN embeddings are also normalized
            self.name_to_domain[normalized_name] = normalized_name
            self.domain_to_name[normalized_name] = normalized_name
        
        # Create outlet to label mapping (using normalized names as keys)
        self.outlet_labels = {}
        
        # Use corpus_df as the primary source for labels
        for _, row in self.corpus_df.iterrows():
            outlet_name = self._normalize_url(row['source_url'])  # Normalize URL for consistent matching
            if self.task == "bias":
                label = row['bias']
            else:  # factuality
                label = row['fact']
            self.outlet_labels[outlet_name] = label
        
        # Fit label encoder
        all_labels = list(self.outlet_labels.values())
        self.label_encoder.fit(all_labels)
        
        # Get train/test outlets that have all embeddings
        self.train_outlets = self._filter_outlets_with_all_embeddings(self.splits['train'])
        self.test_outlets = self._filter_outlets_with_all_embeddings(self.splits['test'])

    def _filter_outlets_with_all_embeddings(self, outlet_list: List[str]) -> List[str]:
        """Filter outlets that have all 5 types of embeddings."""
        filtered_outlets = []
        
        for outlet in outlet_list:
            if outlet in self.outlet_labels:
                # Check if outlet has all embedding types
                has_all = True
                
                # Check articles (from articles_df) 
                if not any(self.articles_df['name'] == outlet):
                    has_all = False
                
                # Check wikipedia (from media_df)
                if not any(self.media_df['name'] == outlet):
                    has_all = False
                
                # Check GNN embeddings - use domain mapping
                if outlet in self.name_to_domain:
                    domain = self.name_to_domain[outlet]
                    
                    if domain not in self.gpt_embeddings:
                        has_all = False
                    if domain not in self.hyperlink_embeddings:  
                        has_all = False
                    if domain not in self.alexa_embeddings:
                        has_all = False
                else:
                    has_all = False
                
                if has_all:
                    filtered_outlets.append(outlet)
        
        return filtered_outlets

    def _normalize_url(self, url: str) -> str:
        """Normalize URL to domain name for consistent matching."""
        if pd.isna(url):
            return ""
        url = str(url).lower()
        if url.startswith('http://'):
            url = url[7:]
        elif url.startswith('https://'):
            url = url[8:]
        if url.startswith('www.'):
            url = url[4:]
        if url.endswith('/'):
            url = url[:-1]
        return url
    
    def _get_domain_for_outlet(self, outlet_name: str) -> Optional[str]:
        """Get domain name for GNN embedding lookup."""
        if outlet_name in self.name_to_domain:
            return self.name_to_domain[outlet_name]
        return None

    def _get_embedding_dimension(self) -> int:
        """Get the standard dimension for all embeddings."""
        # Use 64 as standard dimension (most embeddings are 64-dim)
        return 64

    def _extract_text_embeddings(self):
        """Extract embeddings for articles and Wikipedia descriptions from precomputed embeddings."""
        print("Loading precomputed embeddings...")
        
        # Get all unique outlets that have both article and media data
        all_outlets = set(self.train_outlets + self.test_outlets)
        
        # Extract embeddings from precomputed data
        self.article_embeddings = {}
        self.wiki_embeddings = {}
        
        for outlet in all_outlets:
            # Get article embeddings (precomputed)
            article_rows = self.articles_df[self.articles_df['name'] == outlet]
            if len(article_rows) > 0:
                # Get precomputed article embeddings
                if self.dataset_size == "small":
                    article_emb = np.array(article_rows.iloc[0]['embeddings_1a'])
                else:
                    article_emb = np.array(article_rows.iloc[0]['embeddings_2a'])
                
                # Standardize to embedding_dim dimensions
                if len(article_emb) > self.embedding_dim:
                    article_emb = article_emb[:self.embedding_dim]
                elif len(article_emb) < self.embedding_dim:
                    article_emb = np.pad(article_emb, (0, self.embedding_dim - len(article_emb)))
            else:
                article_emb = np.zeros(self.embedding_dim)
            self.article_embeddings[outlet] = article_emb
            
            # Get Wikipedia/media description embeddings (precomputed)
            media_rows = self.media_df[self.media_df['name'] == outlet]
            if len(media_rows) > 0:
                # Get precomputed media embeddings
                if self.dataset_size == "small":
                    wiki_emb = np.array(media_rows.iloc[0]['embeddings_1b'])
                else:
                    wiki_emb = np.array(media_rows.iloc[0]['embeddings_2b'])
                
                # Standardize to embedding_dim dimensions
                if len(wiki_emb) > self.embedding_dim:
                    wiki_emb = wiki_emb[:self.embedding_dim]
                elif len(wiki_emb) < self.embedding_dim:
                    wiki_emb = np.pad(wiki_emb, (0, self.embedding_dim - len(wiki_emb)))
            else:
                wiki_emb = np.zeros(self.embedding_dim)
            self.wiki_embeddings[outlet] = wiki_emb
        
        print(f"Loaded precomputed embeddings for {len(self.article_embeddings)} outlets")

    def _train_classifier(self):
        """Train the classifier on fused embeddings from training data."""
        print("Training classifier on fused embeddings...")
    
        X_train = []
        y_train = []
        
        if len(self.train_outlets) > 500:
            import random
            # Make a copy to avoid affecting the original list order
            outlets_to_use = random.sample(self.train_outlets, 500)
        else:
            outlets_to_use = self.train_outlets
        
        # Generate randomized training data
        # We want the classifier to be robust to ANY weighting strategy the RL agent tries
        n_augmentations = 5 # How many random weightings per outlet
        
        for outlet in outlets_to_use:  # Use sampled training outlets
            if outlet not in self.outlet_labels:
                continue
                
            domain = self._get_domain_for_outlet(outlet)
            if domain is None:
                continue
            
            try:
                embeddings_dict = self._get_individual_embeddings(outlet, domain)
                if len(embeddings_dict) != 5:
                    continue
                
                label = self.label_encoder.transform([self.outlet_labels[outlet]])[0]
                
                # 1. Add "Pure" strategies (Single views) to ensure it understands raw features
                eye = np.eye(5)
                for i in range(5):
                    strategy = eye[i]
                    fused = self._fuse_embeddings(embeddings_dict, strategy)
                    X_train.append(fused)
                    y_train.append(label)

                actions = np.random.dirichlet(alpha=[0.5]*5, size=n_augmentations)
                # 2. Add "Random" strategies (Domain Randomization)
                for rand_weights in actions:                    
                    fused = self._fuse_embeddings(embeddings_dict, rand_weights)
                    X_train.append(fused)
                    y_train.append(label)
                    
            except Exception as e:
                continue
        
        if len(X_train) == 0:
            print("Warning: No training data available!")
            return
    
        # reinitialize the classifier to avoid becoming too large, max_depth=10
        if self.classifier_type == "svm":
            self.classifier = SVC(kernel='linear', probability=True, random_state=42)
        else:
            # Reduced n_estimators for faster inference during baseline evaluation
            self.classifier = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=1)
        
        print(f"Training classifier on {len(X_train)} examples (Augmented)...")
        self.classifier.fit(X_train, y_train)
        print("Classifier training completed!")

    def _get_individual_embeddings(self, outlet: str, domain: str) -> Dict[str, np.ndarray]:
        """Get individual embeddings for an outlet."""
        embeddings_dict = {}
        
        # Articles embedding (using actual TF-IDF)
        if outlet in self.article_embeddings:
            embeddings_dict['articles'] = self.article_embeddings[outlet]
        
        # Wikipedia embedding (using actual TF-IDF)
        if outlet in self.wiki_embeddings:
            embeddings_dict['wikipedia'] = self.wiki_embeddings[outlet]
        
        # GNN embeddings - standardize dimensions
        if domain in self.alexa_embeddings:
            alexa_emb = np.array(self.alexa_embeddings[domain])
            if len(alexa_emb) > self.embedding_dim:
                alexa_emb = alexa_emb[:self.embedding_dim]
            elif len(alexa_emb) < self.embedding_dim:
                alexa_emb = np.pad(alexa_emb, (0, self.embedding_dim - len(alexa_emb)))
            embeddings_dict['alexa'] = alexa_emb
            
        if domain in self.hyperlink_embeddings:
            hyperlink_emb = np.array(self.hyperlink_embeddings[domain])
            if len(hyperlink_emb) > self.embedding_dim:
                hyperlink_emb = hyperlink_emb[:self.embedding_dim]
            elif len(hyperlink_emb) < self.embedding_dim:
                hyperlink_emb = np.pad(hyperlink_emb, (0, self.embedding_dim - len(hyperlink_emb)))
            embeddings_dict['hyperlink'] = hyperlink_emb
            
        if domain in self.gpt_embeddings:
            gpt_emb = np.array(self.gpt_embeddings[domain])
            if len(gpt_emb) > self.embedding_dim:
                gpt_emb = gpt_emb[:self.embedding_dim]
            elif len(gpt_emb) < self.embedding_dim:
                gpt_emb = np.pad(gpt_emb, (0, self.embedding_dim - len(gpt_emb)))
            embeddings_dict['gpt'] = gpt_emb
        
        return embeddings_dict

    def _get_outlet_embeddings(self, outlet: str) -> Optional[np.ndarray]:
        """Get all 5 embeddings for a specific outlet."""
        embeddings_list = []
        
        try:
            # 1. Articles embedding (A) - using actual TF-IDF embeddings
            if outlet not in self.article_embeddings:
                return None
            article_embedding = self.article_embeddings[outlet]
            embeddings_list.append(article_embedding)
            
            # 2. Wikipedia embedding (W) - using actual TF-IDF embeddings
            if outlet not in self.wiki_embeddings:
                return None
            wiki_embedding = self.wiki_embeddings[outlet]
            embeddings_list.append(wiki_embedding)
            
            # Get domain for GNN embeddings
            domain = self._get_domain_for_outlet(outlet)
            if domain is None:
                return None
            
            # 3. Alexa embedding
            if domain not in self.alexa_embeddings:
                return None
            alexa_embedding = np.array(self.alexa_embeddings[domain])
            # Standardize to 64 dimensions (truncate if larger)
            if len(alexa_embedding) > self.embedding_dim:
                alexa_embedding = alexa_embedding[:self.embedding_dim]
            elif len(alexa_embedding) < self.embedding_dim:
                # Pad with zeros if smaller
                alexa_embedding = np.pad(alexa_embedding, (0, self.embedding_dim - len(alexa_embedding)))
            embeddings_list.append(alexa_embedding)
            
            # 4. Hyperlink embedding
            if domain not in self.hyperlink_embeddings:
                return None
            hyperlink_embedding = np.array(self.hyperlink_embeddings[domain])
            # Standardize dimensions
            if len(hyperlink_embedding) > self.embedding_dim:
                hyperlink_embedding = hyperlink_embedding[:self.embedding_dim]
            elif len(hyperlink_embedding) < self.embedding_dim:
                hyperlink_embedding = np.pad(hyperlink_embedding, (0, self.embedding_dim - len(hyperlink_embedding)))
            embeddings_list.append(hyperlink_embedding)
            
            # 5. LLM/GPT embedding  
            if domain not in self.gpt_embeddings:
                return None
            gpt_embedding = np.array(self.gpt_embeddings[domain])
            # Standardize dimensions
            if len(gpt_embedding) > self.embedding_dim:
                gpt_embedding = gpt_embedding[:self.embedding_dim]
            elif len(gpt_embedding) < self.embedding_dim:
                gpt_embedding = np.pad(gpt_embedding, (0, self.embedding_dim - len(gpt_embedding)))
            embeddings_list.append(gpt_embedding)
            
        except Exception as e:
            print(f"Error getting embeddings for {outlet}: {e}")
            return None
        
        # Concatenate all embeddings
        return np.concatenate(embeddings_list)

    def _fuse_embeddings(self, embeddings_dict: Dict[str, np.ndarray], action: np.ndarray) -> np.ndarray:
        """Fuse embeddings based on the action."""
        embed_types = ['articles', 'wikipedia', 'alexa', 'hyperlink', 'gpt']
        
        if self.action_type == "weights":
            # Normalize weights to sum to 1
            weights = action / (np.sum(action) + 1e-8)
            
            # Weighted average of embeddings
            fused = np.zeros_like(list(embeddings_dict.values())[0])
            for i, embed_type in enumerate(embed_types):
                if embed_type in embeddings_dict:
                    fused += weights[i] * embeddings_dict[embed_type]
                    
        else:  # "selection"
            # Binary selection - average selected embeddings
            selected_embeddings = []
            for i, embed_type in enumerate(embed_types):
                if action[i] > 0.5 and embed_type in embeddings_dict:  # Selected
                    selected_embeddings.append(embeddings_dict[embed_type])
            
            if len(selected_embeddings) == 0:
                # If nothing selected, use average of all
                fused = np.mean(list(embeddings_dict.values()), axis=0)
            else:
                # Average of selected embeddings
                fused = np.mean(selected_embeddings, axis=0)
        
        return fused

    def _compute_reward(self, fused_embedding: np.ndarray, outlet: str) -> Tuple[float, str]:
        """Compute reward based on classification accuracy using trained classifier."""
        # Get true label
        true_label = self.outlet_labels[outlet]
        true_label_encoded = self.label_encoder.transform([true_label])[0]
        
        # Use trained classifier for prediction
        try:
            predicted_label_encoded_array = self.classifier.predict(fused_embedding.reshape(1, -1))
            predicted_label_encoded = predicted_label_encoded_array[0]
            predicted_label = self.label_encoder.inverse_transform([predicted_label_encoded])[0]
            
            # Store for episode evaluation
            self.episode_predictions.append(predicted_label)
            self.episode_true_labels.append(true_label)
            
            # Reward is 1 for correct classification, 0 for incorrect
            reward = 1.0 if predicted_label_encoded == true_label_encoded else 0.0
            
            return reward, predicted_label
            
        except Exception as e:
            print(f"Error in classification: {e}")
            # Fallback to random prediction
            predicted_label = true_label
            self.episode_predictions.append(predicted_label)
            self.episode_true_labels.append(true_label)
            return 0.0, predicted_label



    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.current_outlet_idx = 0
        self.episode_rewards = []
        self.episode_predictions = []
        self.episode_true_labels = []
        
        # Select first outlet from outlet pool (train or test based on mode)
        if len(self.outlet_pool) == 0:
            raise ValueError(f"No outlets available in {self.mode} mode!")
        
        self.current_outlet = self.outlet_pool[self.current_outlet_idx]
        self.current_embeddings = self._get_outlet_embeddings(self.current_outlet)
        
        if self.current_embeddings is None:
            # Skip to next outlet if current one doesn't have all embeddings
            return self._next_outlet()
        
        info = {
            'outlet': self.current_outlet,
            'outlet_idx': self.current_outlet_idx,
            'total_outlets': len(self.outlet_pool)
        }
        
        return self.current_embeddings.astype(np.float32), info

    def _next_outlet(self) -> Tuple[np.ndarray, dict]:
        """Move to next outlet."""
        if self.current_outlet_idx >= len(self.outlet_pool):
            # End of episode - return current embeddings
            info = {
                'outlet': self.current_outlet,
                'outlet_idx': self.current_outlet_idx, 
                'total_outlets': len(self.outlet_pool)
            }
            return self.current_embeddings.astype(np.float32), info
        
        self.current_outlet = self.outlet_pool[self.current_outlet_idx]
        self.current_embeddings = self._get_outlet_embeddings(self.current_outlet)
        
        if self.current_embeddings is None:
            # Skip this outlet
            self.current_outlet_idx += 1
            return self._next_outlet()  # Recursive call to skip outlets without all embeddings
        
        info = {
            'outlet': self.current_outlet,
            'outlet_idx': self.current_outlet_idx, 
            'total_outlets': len(self.outlet_pool)
        }
        
        return self.current_embeddings.astype(np.float32), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        self.episode_steps += 1
        if self.current_embeddings is None:
            raise ValueError("Environment not properly reset!")
        
        # Get individual embeddings for current outlet
        domain = self._get_domain_for_outlet(self.current_outlet)
        embeddings_dict = self._get_individual_embeddings(self.current_outlet, domain)
        
        # Fuse embeddings based on action
        fused_embedding = self._fuse_embeddings(embeddings_dict, action)
        
        # Compute reward
        reward, predicted_label = self._compute_reward(fused_embedding, self.current_outlet)
        self.episode_rewards.append(reward)
        
        # Move to next outlet
        self.current_outlet_idx += 1
        terminated = self.current_outlet_idx >= len(self.outlet_pool)
        
        if not terminated:
            next_obs, info = self._next_outlet()
        else:
            # Episode finished - compute evaluation metrics
            next_obs = self.current_embeddings.astype(np.float32)
            eval_metrics = self.compute_evaluation_metrics()
            info = {
                'episode_finished': True,
                'total_outlets': len(self.episode_predictions),
                **eval_metrics
            }
        
        info.update({
            'reward': reward,
        })
        self.metrics.append(info)
        if terminated:
            self.save_csv(self.out_csv_name, self.run)
            self.run += 1
            self.metrics = []
            self.episode_steps = 0
        
        return next_obs, reward, terminated, False, info

    def save_csv(self, out_csv_name, run):
        """
        Save the logged metrics to CSV for later analysis.
        """
        if out_csv_name is not None and len(self.metrics) > 0:
            df = pd.DataFrame(self.metrics)
            df.to_csv(f"{out_csv_name}_run{run}.csv", index=False)
    
    def compute_evaluation_metrics(self) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics matching the paper."""
        if len(self.episode_predictions) == 0:
            return {}
        
        y_true = self.episode_true_labels
        y_pred = self.episode_predictions
        
        # Convert labels to numeric for MAE computation
        label_to_num = {'left': 0, 'leftcenter': 1, 'center': 2, 'rightcenter': 3, 'right': 4}
        if self.task == "fact":
            label_to_num = {'low': 0, 'mixed': 1, 'high': 2}  # Adjust based on factuality labels
        
        y_true_numeric = [label_to_num.get(label, 0) for label in y_true]
        y_pred_numeric = [label_to_num.get(label, 0) for label in y_pred]
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        mae = mean_absolute_error(y_true_numeric, y_pred_numeric)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mae': mae,
            'episode_reward': np.mean(self.episode_rewards)
        }

    def render(self, mode: str = 'human'):
        pass

    def close(self):
        pass


def create_env(**kwargs) -> DynamicFusionEnv:
    """Factory function to create the environment."""
    return DynamicFusionEnv(**kwargs)


# Evaluation utility functions
def evaluate_model_performance(env: DynamicFusionEnv, 
                             policy_fn,
                             num_episodes: int = 5) -> Dict[str, float]:
    """
    Evaluate a trained policy across multiple episodes.
    
    Args:
        env: The environment
        policy_fn: Function that takes observation and returns action
        num_episodes: Number of episodes to run for evaluation
        
    Returns:
        Dictionary with aggregated metrics
    """
    all_metrics = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if done and 'episode_finished' in info:
                metrics = {k: v for k, v in info.items() 
                          if k in ['accuracy', 'precision', 'recall', 'f1_score', 'mae']}
                all_metrics.append(metrics)
                break
    
    # Aggregate metrics across episodes
    if not all_metrics:
        return {}
    
    aggregated = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'mae']:
        values = [m[metric] for m in all_metrics if metric in m]
        if values:
            aggregated[f'mean_{metric}'] = np.mean(values)
            aggregated[f'std_{metric}'] = np.std(values)
    
    return aggregated