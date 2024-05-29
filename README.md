# Local RAG Model Using "One Hundred Years of Solitude"

This project implements a local Retrieval-Augmented Generation (RAG) model based on the novel "One Hundred Years of Solitude." The model uses the `sentence_transformers` model `all-mpnet-base-v2` for embedding and Google's open-source `google/gemma-2b-it` and `google/flan-t5-large` language models for text generation.

## Repository Structure

- **RAG_one_hundred_years_of_solitude.ipynb**: Jupyter notebook for data preprocessing, model training, and inference.

## Requirements

To install the necessary packages, run the following commands:
pip install sentence-transformers
pip install transformers
pip install torch
pip install numpy
pip install pandas
pip install faiss-cpu
pip install fitz
pip install PyMuPDF
pip install tqdm
pip install accelerate
pip install bitsandbytes
pip install flash-attn --no-build-isolation

## Usage
### Data Preprocessing
The notebook preprocesses text data from "One Hundred Years of Solitude" to create a corpus suitable for the RAG model.

### Model Training
The notebook trains the embedding model and the RAG model using the preprocessed data.

### Inference
The notebook demonstrates how to generate text based on user queries using the trained RAG model.

### Fine-Tuning Parameters
The RAG model can be fine-tuned using the following parameters:

- **Chunk Size of the Corpus**: Determines the size of text chunks used for embedding.
- **Number of Relevant Data Retrieved**: Specifies how many relevant pieces of data are retrieved from the corpus for generation.
- **Embedding Model**: Allows selecting different models for creating text embeddings.
- **LLM Model**: Enables the use of various language models for text generation.
- **Indexing Method**: The model can be fine-tuned using different indexing methods such as dot product or FAISS for efficient similarity search.

## Limitations
The performance of the model is limited due to the available compute power. Enhancing compute resources can significantly improve model performance.

## Models Used
- Embedding Model: sentence-transformers/all-mpnet-base-v2
- Language Models: google/gemma-2b-it, google/flan-t5-large

## Acknowledgments
This project utilizes the following open-source models and libraries:

- Sentence Transformers
- Transformers by Hugging Face
- Google's GEMMA and FLAN-T5 Models
- FAISS


