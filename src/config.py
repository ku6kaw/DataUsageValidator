import os

# Scopus API Configuration
SCOPUS_API_KEY = "YOUR_API_KEY" # Replace with your actual API key
SCOPUS_BASE_URL = "https://api.elsevier.com/content/search/scopus"
SCOPUS_QUERY_DATA_PAPERS = "DOCTYPE(dp)"
SCOPUS_FULLTEXT_API_EID_URL = "https://api.elsevier.com/content/article/eid/"
SCOPUS_FULLTEXT_API_DOI_URL = "https://api.elsevier.com/content/article/doi/"

# LLM Configuration
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY" # Replace with your actual Gemini API key
GEMINI_MODEL_NAME = 'gemini-1.5-flash'
PROMPT_FILE_ZERO_SHOT_ABSTRACT = 'prompts/zero_shot_abstract.txt'
PROMPT_FILE_ZERO_SHOT_FULLTEXT = 'prompts/zero_shot_fulltext.txt'
PROMPT_FILE_FEW_SHOT_COT_FULLTEXT = 'prompts/few_shot_cot_fulltext.txt'


# Data Paths
OUTPUT_DIR_PROCESSED = 'data/processed'
OUTPUT_FILE_DATA_PAPERS = os.path.join(OUTPUT_DIR_PROCESSED, 'data_papers.csv')
OUTPUT_FILE_CITING_PAPERS_RAW = os.path.join(OUTPUT_DIR_PROCESSED, 'citing_papers_raw.csv')
OUTPUT_FILE_CITING_PAPERS_WITH_PATHS = os.path.join(OUTPUT_DIR_PROCESSED, 'citing_papers_with_paths.csv')
OUTPUT_FILE_ANNOTATION_TARGET_LIST = os.path.join('data/ground_truth', 'annotation_target_list.csv')
OUTPUT_FILE_SAMPLES_WITH_TEXT = os.path.join(OUTPUT_DIR_PROCESSED, 'samples_with_text.csv')
OUTPUT_FILE_FEATURES_FOR_EVALUATION = os.path.join(OUTPUT_DIR_PROCESSED, 'features_for_evaluation.csv')
OUTPUT_FILE_PREDICTION_LLM = os.path.join(OUTPUT_DIR_PROCESSED, 'prediction_llm.csv')

RESULTS_DIR = 'results'
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')

XML_OUTPUT_DIR = 'data/raw/fulltext/'
