### 📚 Streamlit Q&A Application with LangChain and Pinecone (PDF, Web Content)

Welcome to an interactive Q&A system using Streamlit! This app uses LangChain and Pinecone to answer questions from PDFs and web content, with advanced preprocessing and OpenAI embeddings. Get precise answers with a cutting-edge large language model (LLM). 🌐🤖

Explore your documents and web content effortlessly! 📄🔍✨

### Installation

Clone the repository to your local machine:

```bash
git clone git@github.com:dirusanov/langchain-tutorial.git
cd langchain-tutorial
```

Install the required packages using pip:

```bash
pip install -r requirements.txt
````
Create a .env file in the root directory and add your Pinecone and OpenAI API credentials:

```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
OPENAI_API_KEY=your_openai_api_key
```

### Directory Structure

Place the PDF documents you want to process in the ./data directory. Ensure the directory follows this structure:

```bash
langchain-tutorial/
├── data/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── main.py
└── .env
```

### Usage

To run the Streamlit application, execute the following command in your terminal:

```bash
streamlit run main.py
```

This command will start the Streamlit server, and you can interact with the application through your web browser.

### License

This project is licensed under the MIT License.
