# Interlinking Analysis and Topic Clustering for WordPress Content

## Description
This Python script performs topic clustering and interlinking analysis for WordPress websites. Using BERTopic and SentenceTransformers, the script processes posts and pages retrieved via WordPress REST API, cleans the content, and clusters it into topics. Additionally, it checks for internal linking issues within the same topic cluster and generates a comprehensive HTML report.

## Features
- Extracts WordPress posts and pages via REST API.
- Cleans content by removing HTML tags, punctuation, and stopwords (supports Italian).
- Generates sentence embeddings using SentenceTransformers.
- Performs topic clustering with BERTopic.
- Identifies interlinking issues within clusters of related content.
- Creates interactive HTML visualizations for:
  - Topic hierarchy.
  - Topic distribution.
  - Document-topic relationships.
- Generates an HTML interlinking report with suggestions for improving internal linking.

## Prerequisites
- Ensure you have Python 3.7 or later installed.
- Install required Python libraries:
pip install os requests json urllib bs4 nltk spacy bertopic sentence-transformers plotly networkx pandas

# Download additional data for NLTK and spaCy:
python -m nltk.downloader punkt stopwords
python -m spacy download it_core_news_lg

## Usage
# 1. Activate your Python environment:
.\amb\Scripts\activate

# 2. Edit the 'domain' variable in the script to specify your WordPress domain:
domain = 'www.example.com'

# 3. Run the script to:
#    - Fetch content from WordPress REST API.
#    - Perform topic clustering and generate embeddings.
#    - Analyze internal linking within clusters.
#    - Generate HTML reports for visualizations and interlinking suggestions.

# 4. Open the generated HTML reports in your browser:
#    - Topic map: 'topics_map.html'
#    - Topic barchart: 'topics_barchart.html'
#    - Topic hierarchy: 'topics_hierarchy.html'
#    - Topic-document relationships: 'topics_docs.html'
#    - Interlinking report: 'yourdomain_interlinking_report.html'

## Output
# Interactive Visualizations:
# - Topics and their keywords.
# - Distribution of topics.
# - Relationships between topics and documents.

# Interlinking Report:
# - Identifies missing internal links within topic clusters.
# - Provides actionable suggestions for improving interlinking.

## Customization
# Modify the BERTopic model parameters for granularity:
topic_model = BERTopic(
    language="italian",
    min_topic_size=2,  # Adjust cluster size
    calculate_probabilities=True
)

# Replace the SentenceTransformer model to customize for your language or use case:
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # Default model
# Alternative models:
# model = SentenceTransformer('xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
# model = SentenceTransformer('sentence-transformers/LaBSE')

## Notes
# Ensure your WordPress REST API endpoints are accessible and return valid JSON data.
# Outliers (documents not assigned to any topic) are reported separately.

