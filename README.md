
# 💬 Semantic Quote Retrieval System with RAG

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to semantically retrieve meaningful quotes from the [`Abirate/english_quotes`](https://huggingface.co/datasets/abirate/english_quotes) dataset using a fine-tuned model and interactive Streamlit interface.

🎥 **Demo Video**: [Watch here](https://www.loom.com/share/9b62ea8608c14110b4ca83e0e6ccbb9b?sid=449b9adc-c1df-4992-88f8-50fe4a574180)

---

## 📁 Project Structure

```
quote_rag_system/
├── data/
│   └── processed_quotes.json 
├── models/
│   └── fine_tuned_model/ 
├── src/
│   ├── data_preparation.py 
│   ├── model_training.py 
│   ├── rag_pipeline.py
│   ├── evaluation.py 
│   └── streamlit_app.py 
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Setup Virtual Environment

```bash
python -m venv venv
```

#### Activate the environment:

- On **Windows**:
  ```bash
  venv\Scripts\activate
  ```

- On **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### 4. Prepare Data

```bash
cd src
python data_preparation.py
```

### 5. Fine-Tune Embedding Model

```bash
python model_training.py
```

### 6. Build RAG Pipeline

```bash
python rag_pipeline.py
```

### 7. Evaluate Retrieval Performance

```bash
python evaluation.py
```

### 8. Launch Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## 🧠 Model Architecture

- **Base Encoder**: `sentence-transformers/all-MiniLM-L6-v2` (fine-tuned)
- **Retriever**: FAISS for fast similarity search
- **Generator**: OpenAI GPT API
- **RAG Integration**: Combines vector retrieval with generation from GPT for contextual responses

---

## ✅ Design Decisions

- **Minimal Preprocessing**: Preserved semantic richness, only removed special characters and duplicates.
- **FAISS Indexing**: Chosen for high speed with ~10k+ quotes.
- **SentenceTransformer Fine-tuning**: Boosted relevance of semantic search results.
- **Streamlit UI**: Easy-to-use and deployable interface.

---

## ⚠️ Challenges

- **Embedding Quality**: Initially poor, improved by fine-tuning on quote dataset.
- **Memory Usage with FAISS**: Tuned vector size and indexing batch.
- **Evaluation using RAGAS**: Required careful formatting of inputs/outputs.
- **OpenAI API Costs**: Optimized prompt sizes and token usage.

---

## 📊 Example

**Query**: "Motivation for hard times"  
**Top Quote Retrieved**:  
> “Hardships often prepare ordinary people for an extraordinary destiny.” – C.S. Lewis

---