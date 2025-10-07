
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import json
from collections import defaultdict
import re

class ChatRecommendationSystem:
    
    def __init__(self):
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
        
    def preprocess_text(self, text):
        
        # Remove quotes
        text = text.strip('"')
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
        print(f"Senders: {senders}")

        if len(senders) >= 2:
            user_a = senders[0]
            user_b = senders[1]
        else:
            raise ValueError("Need at least two different senders in the conversation")
        
        print(f"User A: {user_a}")
        print(f"User B: {user_b}")
   
        self.create_message_pairs(df, user_a, user_b)
        
        self.conversation_data = df
        return df
    
    def create_message_pairs(self, df, user_a, user_b):
       
        print("\nCreating message pairs...")

        conversations = df.groupby('Conversation ID')
        
        for conv_id, conv_df in conversations:
            messages = conv_df.sort_values('Timestamp').reset_index(drop=True)
            
            for i in range(len(messages)):
                current_sender = messages.loc[i, 'Sender']
                current_message = messages.loc[i, 'Message']
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

        print("Vectorizing messages...")
        self.input_vectors = self.vectorizer.fit_transform(combined_inputs)
        
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print("Model trained successfully!")
    
    def predict_reply(self, user_b_message, context="", top_k=3):

        user_b_message = self.preprocess_text(user_b_message)
        

        combined_input = f"{context} [SEP] {user_b_message}"

        input_vector = self.vectorizer.transform([combined_input])

        similarities = cosine_similarity(input_vector, self.input_vectors)[0]

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        predictions = []
        for idx in top_indices:
            predictions.append({
                'reply': self.user_a_messages[idx],
                'similarity': float(similarities[idx]),
                'context': self.message_pairs[idx]['context'],
                'original_input': self.message_pairs[idx]['user_b_input']
            })
        
        return predictions
    
    def evaluate(self):
        """Evaluate the model using various metrics"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        if len(self.message_pairs) < 2:
            print("Not enough data for evaluation")
            return

        correct = 0
        total = len(self.message_pairs)
        
        for i, pair in enumerate(self.message_pairs):

            predictions = self.predict_reply(pair['user_b_input'], pair['context'], top_k=3)

            predicted_replies = [p['reply'] for p in predictions]
            if pair['user_a_reply'] in predicted_replies:
                correct += 1
        
        accuracy = (correct / total) * 100
        print(f"\nTop-3 Accuracy: {accuracy:.2f}%")
        print(f"Correct predictions: {correct}/{total}")

        all_similarities = []
        for pair in self.message_pairs:
            predictions = self.predict_reply(pair['user_b_input'], pair['context'], top_k=1)
            if predictions:
                all_similarities.append(predictions[0]['similarity'])
        
        avg_similarity = np.mean(all_similarities)
        print(f"Average similarity score: {avg_similarity:.4f}")
        
        return {
            'accuracy': accuracy,
            'avg_similarity': avg_similarity,
            'total_pairs': total
        }
    
    def save_model(self, model_path='Model.joblib'):
        """Save the trained model"""
        model_data = {
            'vectorizer': self.vectorizer,
            'message_pairs': self.message_pairs,
            'user_a_messages': self.user_a_messages,
            'user_b_messages': self.user_b_messages,
            'input_vectors': self.input_vectors,
            'context_window': self.context_window
        }
        joblib.dump(model_data, model_path)
        print(f"\nModel saved to {model_path}")
    
    def load_model(self, model_path='Model.joblib'):
        """Load a saved model"""
        model_data = joblib.load(model_path)
        self.vectorizer = model_data['vectorizer']
        self.message_pairs = model_data['message_pairs']
        self.user_a_messages = model_data['user_a_messages']
        self.user_b_messages = model_data['user_b_messages']
        self.input_vectors = model_data['input_vectors']
        self.context_window = model_data['context_window']
        print(f"Model loaded from {model_path}")


def main():
    """Main training and evaluation pipeline"""
    print("="*60)
    print("CHAT REPLY RECOMMENDATION SYSTEM")
    print("="*60)

    chat_system = ChatRecommendationSystem()
  
    data_path = 'conversation_data.csv'
    try:
        chat_system.load_and_preprocess_data(data_path)
    except FileNotFoundError:
        print(f"File not found: {data_path}")
        print("Trying Excel file...")
        data_path = 'attached_assets/conversationfile_1759843402038.xlsx'
        chat_system.load_and_preprocess_data(data_path)
    

    chat_system.train()

    metrics = chat_system.evaluate()

    chat_system.save_model('Model.joblib')

    print("\n" + "="*60)
    print("DEMO PREDICTIONS")
    print("="*60)

    test_messages = [
        "Any plans for Saturday?",
        "Want to join?",
        "What time?"
    ]
    
    for msg in test_messages:
        print(f"\nUser B: \"{msg}\"")
        predictions = chat_system.predict_reply(msg, top_k=3)
        print("\nTop 3 Predicted Replies from User A:")
        for i, pred in enumerate(predictions, 1):
            print(f"  {i}. \"{pred['reply']}\" (similarity: {pred['similarity']:.4f})")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return chat_system, metrics


if __name__ == "__main__":
    system, metrics = main()
