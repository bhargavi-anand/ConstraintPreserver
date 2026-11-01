# ConstraintPreserver

### A Constraint-Preserving Document Retrieval and Question-Answering System

ConstraintPreserver is a Python-based document retrieval and question-answering system designed to retrieve relevant information from documents while preserving important constraints, conditions, and contextual details.

The project focuses on improving the reliability of document-based question answering by combining document preprocessing, intelligent chunking, semantic search, and constraint-aware answer generation.

## 🎯 Project Objectives

* Extract and preprocess information from documents.
* Divide documents into meaningful chunks for efficient retrieval.
* Generate semantic embeddings for document chunks.
* Perform similarity-based retrieval using FAISS.
* Identify and preserve important constraints from retrieved information.
* Generate contextually relevant answers based on retrieved documents.

## ⚙️ Technology Stack

| Technology            | Purpose                                   |
| --------------------- | ----------------------------------------- |
| Python                | Core programming language                 |
| FAISS                 | Vector similarity search                  |
| Sentence Transformers | Semantic embeddings                       |
| JSON                  | Document and metadata storage             |
| NLP                   | Text processing and information retrieval |

## 🏗️ Project Architecture

The project follows a modular architecture consisting of the following stages:

**1. Document Preprocessing**

* Document extraction and cleaning.
* Text normalization.
* Document validation.

**2. Document Chunking**

* Dividing documents into smaller chunks.
* Maintaining relevant contextual information.
* Generating structured document metadata.

**3. Vector Indexing**

* Generating semantic embeddings.
* Storing embeddings in a FAISS index.
* Maintaining associated chunk metadata.

**4. Semantic Retrieval**

* Converting user queries into embeddings.
* Searching for relevant document chunks.
* Retrieving the most relevant information.

**5. Constraint Extraction**

* Identifying important conditions and restrictions.
* Preserving critical information during answer generation.

**6. Answer Generation**

* Generating answers using retrieved document context.
* Maintaining consistency with the original document information.

## 📁 Project Structure

```text
ConstraintPreserver/
│
├── data/
│   ├── processed/
│   └── index/
│
├── src/
│   ├── preprocessing/
│   │   ├── build_documents_json.py
│   │   ├── clean_documents.py
│   │   └── chunk_documents.py
│   │
│   ├── validation/
│   │   └── verify_cleaned_documents.py
│   │
│   ├── retrieval/
│   │   ├── build_index.py
│   │   ├── search.py
│   │   └── answer.py
│   │
│   ├── constraints/
│   │   └── extractor.py
│   │
│   └── generation/
│
├── .gitignore
└── README.md
```

*Note: The structure above represents the main modules; additional files may exist in the repository.*

## 🚀 Key Features

* Modular document processing pipeline.
* Semantic document retrieval.
* FAISS-based vector indexing.
* Metadata-aware document chunking.
* Constraint extraction and preservation.
* Context-based question answering.

## 💻 Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/bhargavi-anand/ConstraintPreserver.git
```

**2. Navigate to the project directory**

```bash
cd ConstraintPreserver
```

**3. Create a virtual environment**

```bash
python -m venv .venv
```

**4. Activate the environment**

Windows:

```bash
.venv\Scripts\activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

## 🔬 Project Workflow

```text
Input Documents
      |
      v
Document Extraction
      |
      v
Text Cleaning & Validation
      |
      v
Document Chunking
      |
      v
Embedding Generation
      |
      v
FAISS Vector Index
      |
      v
Semantic Retrieval
      |
      v
Constraint Extraction
      |
      v
Context-Aware Answer Generation
      |
      v
Final Answer
```

## 📌 Future Enhancements

* Support for additional document formats.
* Improved constraint extraction techniques.
* Enhanced retrieval accuracy.
* Evaluation using benchmark datasets.
* Integration of advanced language models.
* Improved handling of complex multi-condition queries.

## 👩‍💻 Contributors

Developed as a project exploring Natural Language Processing, semantic search, document retrieval, and constraint-aware question answering.

---

**Project:** ConstraintPreserver
**Repository:** https://github.com/bhargavi-anand/ConstraintPreserver
