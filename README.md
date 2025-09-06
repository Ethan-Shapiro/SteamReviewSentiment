# Steam Review Sentiment Analysis

A big data analytics project that performs sentiment analysis on Steam game reviews using PySpark and Natural Language Processing (NLP) techniques. This project analyzes over 113 million Steam reviews to extract insights about game sentiment, player behavior, and review patterns.

## 🎯 Project Overview

This project demonstrates large-scale data processing and sentiment analysis on Steam game reviews. It uses Apache Spark for distributed computing and Spark NLP for advanced text processing, including sentiment analysis and named entity recognition.

### Key Features

- **Big Data Processing**: Handles 113+ million Steam reviews using PySpark
- **Sentiment Analysis**: Performs unsupervised sentiment analysis using TextBlob
- **Named Entity Recognition**: Extracts entities from reviews using Spark NLP
- **Data Visualization**: Creates word clouds and sentiment distributions
- **Memory Optimization**: Implements efficient data partitioning and memory management
- **Multi-language Support**: Filters and analyzes English reviews from the dataset

## 📊 Dataset

The project works with a comprehensive Steam reviews dataset (`all_reviews.csv`) containing:
- **113,885,601** total reviews
- **51,544,179** English reviews (after filtering)
- **96,492** distinct games
- Review metadata including playtime, votes, timestamps, and user information

### Data Schema

The dataset includes the following fields:
- Game information (app ID, game name)
- Author details (Steam ID, games owned, playtime statistics)
- Review content and metadata (text, language, timestamps)
- Community engagement (votes up, funny votes, comments)
- Purchase information (Steam purchase, received for free, early access)

## 🛠 Technology Stack

- **Apache Spark 3.2.3**: Distributed data processing
- **PySpark**: Python API for Spark
- **Spark NLP 5.5.0**: Advanced NLP processing and entity recognition
- **TextBlob**: Sentiment analysis
- **Python 3.12**: Core programming language
- **Jupyter Notebook**: Interactive development environment
- **Data Visualization**: Matplotlib, Seaborn, WordCloud
- **Data Processing**: Pandas, NLTK

## 🚀 Getting Started

### Prerequisites

- Python 3.8+ (tested with Python 3.12)
- Apache Spark ≤ 3.2
- Hadoop ≤ 3.0
- Java 8 or 11 (required for Spark)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd SteamReviewSentiment
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv env
   # On Windows:
   env\Scripts\activate
   # On macOS/Linux:
   source env/bin/activate
   ```

3. **Install required packages**:
   ```bash
   pip install pyspark==3.2.3
   pip install spark-nlp
   pip install textblob
   pip install matplotlib seaborn wordcloud
   pip install pandas numpy
   pip install nltk
   pip install jupyter
   ```

4. **Download NLTK data**:
   ```python
   import nltk
   nltk.download('stopwords')
   ```

### Configuration

The project requires specific Spark configuration for optimal performance:

```python
spark = SparkSession.builder \
    .config("spark.driver.host", "localhost") \
    .appName("Steam Reviews Analysis") \
    .config("spark.ui.port", "4050") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "2000m") \
    .config("spark.jsl.settings.pretrained.cache_folder", "sample_data/pretrained") \
    .config("spark.jsl.settings.storage.cluster_tmp_dir", "sample_data/storage") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.0") \
    .getOrCreate()
```

## 📈 Usage

### Running the Analysis

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the main analysis notebook**:
   - `SteamReviewsAnalysis.ipynb`

3. **Run the cells sequentially** to:
   - Load and explore the dataset
   - Perform data cleaning and preprocessing
   - Conduct sentiment analysis
   - Generate visualizations and word clouds
   - Extract named entities from reviews

### Key Analysis Steps

1. **Data Loading & Schema Optimization**: Loads CSV data with optimized schema for memory efficiency
2. **Data Filtering**: Removes non-English reviews and games with fewer than 200 reviews
3. **Partitioning**: Optimizes data distribution across Spark partitions for performance
4. **Sentiment Analysis**: Applies TextBlob sentiment analysis using User Defined Functions (UDFs)
5. **Entity Recognition**: Uses Spark NLP pipeline for named entity extraction
6. **Visualization**: Creates sentiment distributions and word clouds for positive/negative reviews

## 📊 Results

The project generates several insights:

- **Language Distribution**: Analysis of review languages with English comprising ~45% of total reviews
- **Sentiment Patterns**: Distribution of sentiment scores across game reviews
- **Entity Analysis**: Word clouds highlighting key entities in positive vs. negative reviews
- **Game Statistics**: Average playtime, vote patterns, and review metrics by game

### Sample Outputs

- Sentiment distribution histograms
- Word clouds for positive and negative reviews
- Language distribution bar charts
- Game-specific statistics and insights

## 🔧 Performance Optimizations

The project implements several performance optimizations:

- **Memory Management**: Explicit cache management and garbage collection
- **Data Partitioning**: Strategic repartitioning based on data access patterns
- **Schema Optimization**: Using appropriate data types to minimize memory usage
- **Lazy Evaluation**: Leveraging Spark's lazy evaluation for efficient processing
- **Partition Sizing**: Optimal partition count based on available cores (2-4 partitions per core)

## 📁 Project Structure

```
SteamReviewSentiment/
├── SteamReviewsAnalysis.ipynb    # Main analysis notebook
├── all_reviews.csv               # Steam reviews dataset (200MB+)
├── reviews_with_sentiment.parquet # Processed data with sentiment scores
├── env/                          # Python virtual environment
├── sample_data/                  # Spark NLP model cache and storage
│   ├── pretrained/              # Pre-trained NLP models
│   └── storage/                 # Temporary storage for Spark NLP
└── README.md                    # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Steam for providing the review data
- John Snow Labs for Spark NLP
- Apache Spark community for the distributed computing framework
- TextBlob for sentiment analysis capabilities

## 📞 Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This project requires significant computational resources due to the large dataset size. Ensure adequate memory (8GB+ RAM recommended) and storage space (500MB+ for models and cache) before running the analysis.
