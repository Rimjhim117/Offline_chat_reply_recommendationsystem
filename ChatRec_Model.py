import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import json
from collections import defaultdict
import re
import os

class ChatRecommendationSystem:
    
    def __init__(self, use_semantic=True, semantic_weight=0.7):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            min_df=1,
            lowercase=True,
            stop_words='english'
        )
        self.conversation_data = None
        self.user_a_messages = []
        self.user_b_messages = []
        self.message_pairs = []
        self.context_window = 3  
        
        # Semantic parameters
        self.use_semantic = use_semantic
        self.semantic_weight = semantic_weight
        self.semantic_model = None
        self.semantic_embeddings = None
        
        if self.use_semantic:
            try:
                from sentence_transformers import SentenceTransformer
                print("Loading SentenceTransformer ('all-MiniLM-L6-v2')...")
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("SentenceTransformer model loaded successfully!")
            except Exception as e:
                print(f"Warning: Failed to load SentenceTransformer: {e}. Falling back to TF-IDF only.")
                self.use_semantic = False
        
    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        # Remove quotes
        text = text.strip('"').strip("'")
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def load_and_preprocess_data(self, data_path):
        print("Loading conversation data...")
        # Load data
        if data_path.endswith('.xlsx'):
            df = pd.read_excel(data_path)
        else:
            df = pd.read_csv(data_path)
        
        print(f"Loaded {len(df)} messages")
        print(f"Columns: {df.columns.tolist()}")
        
        df['Message'] = df['Message'].apply(self.preprocess_text)
        
        senders = df['Sender'].unique()
        print(f"Senders found in dataset: {senders}")

        # Correctly map User A and User B
        user_a = None
        user_b = None
        
        for s in senders:
            clean_s = str(s).strip().lower()
            if clean_s in ['user a', 'usera', 'a']:
                user_a = s
            elif clean_s in ['user b', 'userb', 'b']:
                user_b = s
                
        if user_a is None or user_b is None:
            if len(senders) >= 2:
                # Find initiator of the first conversation
                sorted_df = df.sort_values(by=['Conversation ID', 'Timestamp']).reset_index(drop=True)
                first_sender = sorted_df.loc[0, 'Sender']
                user_b = first_sender
                user_a = [s for s in senders if s != user_b][0]
            else:
                raise ValueError("Need at least two different senders in the conversation")
                
        print(f"Mapped Senders -> Initiator (User B): '{user_b}', Responder (User A): '{user_a}'")
        
        self.create_message_pairs(df, user_a, user_b)
        self.conversation_data = df
        return df
    
    def create_message_pairs(self, df, user_a, user_b):
        print("\nCreating message pairs...")
        self.message_pairs = []
        
        conversations = df.groupby('Conversation ID')
        
        for conv_id, conv_df in conversations:
            messages = conv_df.sort_values('Timestamp').reset_index(drop=True)
            
            for i in range(len(messages)):
                current_sender = messages.loc[i, 'Sender']
                current_message = messages.loc[i, 'Message']
                
                # If current message is User A's response and there's previous context
                if current_sender == user_a and i > 0:
                    context_start = max(0, i - self.context_window)
                    context_messages = []
                    
                    for j in range(context_start, i):
                        prev_sender = messages.loc[j, 'Sender']
                        prev_message = messages.loc[j, 'Message']
                        context_messages.append(f"{prev_sender}: {prev_message}")
                        
                    user_b_message = messages.loc[i-1, 'Message']
                    user_a_reply = current_message
                    
                    context = " [SEP] ".join(context_messages) if context_messages else ""
                    
                    self.message_pairs.append({
                        'context': context,
                        'user_b_input': user_b_message,
                        'user_a_reply': user_a_reply,
                        'conv_id': conv_id
                    })
        
        print(f"Created {len(self.message_pairs)} message pairs")
        self.user_b_messages = [pair['user_b_input'] for pair in self.message_pairs]
        self.user_a_messages = [pair['user_a_reply'] for pair in self.message_pairs]
    
    def train(self):
        print("\nTraining the model...")
        if not self.message_pairs:
            raise ValueError("No message pairs available. Run load_and_preprocess_data first.")
            
        combined_inputs = []
        for pair in self.message_pairs:
            combined = f"{pair['context']} [SEP] {pair['user_b_input']}"
            combined_inputs.append(combined)

        print("Fitting TF-IDF Vectorizer...")
        self.input_vectors = self.vectorizer.fit_transform(combined_inputs)
        print(f"TF-IDF Vocabulary size: {len(self.vectorizer.vocabulary_)}")

        if self.use_semantic and self.semantic_model is not None:
            print("Encoding combined inputs using SentenceTransformer...")
            self.semantic_embeddings = self.semantic_model.encode(combined_inputs, show_progress_bar=False)
            print("Sentence embeddings shape:", self.semantic_embeddings.shape)
            
        print("Model trained successfully!")
        
    def predict_reply(self, user_b_message, context="", top_k=3, semantic_weight=None):
        if semantic_weight is None:
            semantic_weight = self.semantic_weight
            
        user_b_message = self.preprocess_text(user_b_message)
        combined_input = f"{context} [SEP] {user_b_message}"

        # TF-IDF Cosine Similarity
        input_vector = self.vectorizer.transform([combined_input])
        tfidf_similarities = cosine_similarity(input_vector, self.input_vectors)[0]

        # Semantic Similarity
        if self.use_semantic and self.semantic_embeddings is not None and self.semantic_model is not None:
            query_embedding = self.semantic_model.encode([combined_input], show_progress_bar=False)
            semantic_similarities = cosine_similarity(query_embedding, self.semantic_embeddings)[0]
            
            # Combine scores
            similarities = semantic_weight * semantic_similarities + (1.0 - semantic_weight) * tfidf_similarities
        else:
            semantic_similarities = np.zeros_like(tfidf_similarities)
            similarities = tfidf_similarities

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        predictions = []
        for idx in top_indices:
            predictions.append({
                'reply': self.user_a_messages[idx],
                'similarity': float(similarities[idx]),
                'tfidf_similarity': float(tfidf_similarities[idx]),
                'semantic_similarity': float(semantic_similarities[idx]),
                'context': self.message_pairs[idx]['context'],
                'original_input': self.message_pairs[idx]['user_b_input']
            })
        
        return predictions

    def evaluate(self):
        """Evaluate performance of TF-IDF, Semantic, and Hybrid models on training set and cross-validation"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        total = len(self.message_pairs)
        if total < 2:
            print("Not enough data for evaluation")
            return {}

        results = {}
        for mode in ['TF-IDF Only', 'Semantic Only', 'Hybrid (0.7 Sem / 0.3 TF-IDF)']:
            if mode == 'TF-IDF Only':
                w = 0.0
            elif mode == 'Semantic Only':
                w = 1.0
            else:
                w = self.semantic_weight
                
            if mode != 'TF-IDF Only' and not self.use_semantic:
                continue

            # 1. Training Set Accuracy
            train_correct = 0
            all_sims = []
            for pair in self.message_pairs:
                predictions = self.predict_reply(pair['user_b_input'], pair['context'], top_k=3, semantic_weight=w)
                predicted_replies = [p['reply'] for p in predictions]
                if pair['user_a_reply'] in predicted_replies:
                    train_correct += 1
                if predictions:
                    all_sims.append(predictions[0]['similarity'])
            
            train_acc = (train_correct / total) * 100
            avg_sim = np.mean(all_sims) if all_sims else 0.0

            # 2. Leave-One-Out Cross-Validation
            loo_correct = 0
            combined_inputs = [f"{p['context']} [SEP] {p['user_b_input']}" for p in self.message_pairs]
            
            for val_idx in range(total):
                train_indices = [idx for idx in range(total) if idx != val_idx]
                train_combined = [combined_inputs[idx] for idx in train_indices]
                train_replies = [self.user_a_messages[idx] for idx in train_indices]
                
                # Fit TF-IDF on training subset
                sub_vectorizer = TfidfVectorizer(
                    max_features=1000,
                    ngram_range=(1, 3),
                    min_df=1,
                    lowercase=True,
                    stop_words='english'
                )
                sub_tfidf = sub_vectorizer.fit_transform(train_combined)
                val_tfidf_vec = sub_vectorizer.transform([combined_inputs[val_idx]])
                val_tfidf_sims = cosine_similarity(val_tfidf_vec, sub_tfidf)[0]
                
                if mode != 'TF-IDF Only' and self.use_semantic and self.semantic_embeddings is not None:
                    # Semantic search subset
                    sub_semantic = self.semantic_embeddings[train_indices]
                    val_sem_vec = self.semantic_embeddings[val_idx:val_idx+1]
                    val_sem_sims = cosine_similarity(val_sem_vec, sub_semantic)[0]
                    
                    sims = w * val_sem_sims + (1.0 - w) * val_tfidf_sims
                else:
                    sims = val_tfidf_sims
                    
                top_3_indices = np.argsort(sims)[-3:][::-1]
                top_3_replies = [train_replies[idx] for idx in top_3_indices]
                
                if self.message_pairs[val_idx]['user_a_reply'] in top_3_replies:
                    loo_correct += 1
                    
            loo_acc = (loo_correct / total) * 100
            
            print(f"\nMode: {mode}")
            print(f"  Training Top-3 Accuracy: {train_acc:.2f}% ({train_correct}/{total})")
            print(f"  Training Avg Similarity: {avg_sim:.4f}")
            print(f"  Leave-One-Out CV Top-3 Accuracy: {loo_acc:.2f}% ({loo_correct}/{total})")
            
            results[mode] = {
                'train_acc': train_acc,
                'avg_similarity': avg_sim,
                'loo_acc': loo_acc
            }
            
        return results
    
    def save_model(self, model_path='Model.joblib'):
        """Save the trained model components"""
        model_data = {
            'vectorizer': self.vectorizer,
            'message_pairs': self.message_pairs,
            'user_a_messages': self.user_a_messages,
            'user_b_messages': self.user_b_messages,
            'input_vectors': self.input_vectors,
            'context_window': self.context_window,
            'semantic_embeddings': self.semantic_embeddings,
            'use_semantic': self.use_semantic,
            'semantic_weight': self.semantic_weight
        }
        joblib.dump(model_data, model_path)
        print(f"\nModel saved to {model_path}")
    
    def load_model(self, model_path='Model.joblib'):
        """Load the trained model components"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
            
        model_data = joblib.load(model_path)
        self.vectorizer = model_data['vectorizer']
        self.message_pairs = model_data['message_pairs']
        self.user_a_messages = model_data['user_a_messages']
        self.user_b_messages = model_data['user_b_messages']
        self.input_vectors = model_data['input_vectors']
        self.context_window = model_data['context_window']
        self.semantic_embeddings = model_data.get('semantic_embeddings', None)
        self.use_semantic = model_data.get('use_semantic', False)
        self.semantic_weight = model_data.get('semantic_weight', 0.7)
        
        if self.use_semantic and self.semantic_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Warning: Failed to load SentenceTransformer: {e}")
                self.use_semantic = False
                
        print(f"Model loaded from {model_path} successfully!")


def main():
    print("="*60)
    print("CHAT REPLY RECOMMENDATION SYSTEM (UPGRADED)")
    print("="*60)

    chat_system = ChatRecommendationSystem(use_semantic=True, semantic_weight=0.7)
  
    data_path = 'conversation_data.csv'
    if not os.path.exists(data_path):
        data_path = 'attached_assets/conversationfile_1759843402038.xlsx'
        
    chat_system.load_and_preprocess_data(data_path)
    chat_system.train()
    metrics = chat_system.evaluate()
    chat_system.save_model('Model.joblib')

    print("\n" + "="*60)
    print("DEMO PREDICTIONS (COMPARING QUERY VARIANTS)")
    print("="*60)

    # Let's test standard and semantically similar queries to prove semantic matching
    test_cases = [
        ("Any plans for Saturday?", "Weekend plans?"),
        ("Want to join?", "Care to come along?"),
        ("What time?", "When should we meet?")
    ]
    
    for standard_q, semantic_q in test_cases:
        print(f"\n--- Testing Query Group ---")
        for q in [standard_q, semantic_q]:
            print(f"\nUser B Input: \"{q}\"")
            predictions = chat_system.predict_reply(q, top_k=3)
            print("Top 3 Predicted Replies from User A:")
            for i, pred in enumerate(predictions, 1):
                print(f"  {i}. \"{pred['reply']}\"")
                print(f"     [Scores] Hybrid: {pred['similarity']:.4f} | Sem: {pred['semantic_similarity']:.4f} | TFIDF: {pred['tfidf_similarity']:.4f}")
    
    print("\n" + "="*60)
    print("TRAINING AND EVALUATION COMPLETE!")
    print("="*60)
    
    return chat_system, metrics


if __name__ == "__main__":
    main()
