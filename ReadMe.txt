================================================================================
OFFLINE SEMANTIC CHAT REPLY RECOMMENDATION SYSTEM (UPGRADED)
================================================================================

PROJECT OVERVIEW:
This is an advanced, offline, context-aware chat reply recommendation system.
It utilizes a Hybrid Matcher combining Sentence-Transformers (for semantic 
meaning and synonym handling) and TF-IDF (for exact vocabulary overlap) to 
accurately predict User A's replies to User B's messages.

Unlike standard keyword-matching bots, this system recognizes user intent even
when different words or phrasings are used (e.g., matching "Weekend plans?"
to "Any plans for Saturday?").

================================================================================
FILES INCLUDED:
================================================================================

1. ChatRec_Model.py      - Main model logic, hybrid similarity, and evaluation
2. app.py                - Streamlit-powered premium web dashboard
3. Model.joblib          - Serialized model, TF-IDF weights, and precomputed embeddings
4. conversation_data.csv - Conversation history dataset (22 messages, 4 conversations)
5. ReadMe.txt            - This documentation file

================================================================================
SYSTEM ARCHITECTURE:
================================================================================

1. AUTOMATIC SENDER RESOLUTION:
   - Analyzes dataset senders to dynamically isolate the initiator (User B) and 
     the responder (User A).
   - Resolves bugs from previous implementations where role assignments were 
     dependent on the row order.

2. CONTEXT-AWARE PACKING:
   - Packages historical conversation exchanges (context window = 3) with the 
     trigger message using a special separation format:
     `[Sender]: [Message] [SEP] ... [SEP] [User B Trigger Message]`

3. HYBRID MATCHING ALGORITHM:
   - Semantic Encoder: Encodes the prompt into a 384-dimensional dense vector 
     using the pre-trained 'all-MiniLM-L6-v2' model.
   - Keyword Matcher: Creates a TF-IDF sparse matrix (1-3 n-grams).
   - Ensemble Score: Combines both scores using a weighted average:
     `Similarity = (0.7 * Semantic_Score) + (0.3 * TFIDF_Score)`
   - Light & Offline: Loads in milliseconds, runs entirely locally, and 
     requires no GPU.

4. SEMANTIC CROSS-VALIDATION:
   - Includes Leave-One-Out (LOO) cross-validation.
   - Evaluates system capability to retrieve contextually appropriate responses
     when the exact sample is withheld, preventing training-set data leakage.

================================================================================
REQUIREMENTS:
================================================================================

- Python 3.10+
- Dependencies:
  - streamlit
  - sentence-transformers
  - scikit-learn
  - pandas
  - numpy
  - joblib
  - matplotlib

To install all dependencies (if running outside this pre-configured environment):
  pip install streamlit sentence-transformers scikit-learn pandas numpy joblib matplotlib

================================================================================
HOW TO RUN:
================================================================================

1. RUN CORE BENCHMARKS & TEST PREDICTIONS:
   
   python ChatRec_Model.py

   This will:
   - Load the dataset and correctly map the senders.
   - Fit the TF-IDF vectorizer and compute Sentence-Transformer embeddings.
   - Run benchmarks for TF-IDF vs. Semantic vs. Hybrid.
   - Print out comparison queries showing the superiority of Semantic matching.
   - Serialize the trained model to Model.joblib.

2. RUN THE STREAMLIT WEB DASHBOARD:
   
   streamlit run app.py

   This launches an interactive, custom-styled dashboard in your browser.
   Features include:
   - **Interactive Simulator**: Type a message as User B, set a custom 
     conversation history context, and view real-time ranked recommendations 
     for User A (complete with similarity scores).
   - **CSV/Excel Uploader**: Instantly train the hybrid recommender on any custom
     chat dataset.

================================================================================
DEVELOPMENT NOTES:
================================================================================
- Model Version: 2.0 (Upgraded from simple TF-IDF baseline)
- Pre-trained Embeddings: all-MiniLM-L6-v2 (384 Dimensions)
- Design Guidelines: Clean, structured layout with responsive custom styles

================================================================================
CONTACT: rimjhimsrivastava971@gmail.com
================================================================================
