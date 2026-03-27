from sentence_transformers import SentenceTransformer, util
from duckduckgo_search import DDGS
import advanced_config

class RealTimeValidator:
    def __init__(self, use_sbert=True):
        self.use_sbert = use_sbert
        if self.use_sbert:
            print(f"Loading SBERT model: {advanced_config.SIMILARITY_MODEL_NAME}")
            self.sbert = SentenceTransformer(advanced_config.SIMILARITY_MODEL_NAME)
        self.ddgs = DDGS()
        
    def extract_keywords(self, text):
        """
        Simple extraction of first few words or nouns.
        For production, use SpaCy or NLTK NER. 
        Here we'll use a simplified heuristic for speed.
        """
        words = text.split()
        return " ".join(words[:15])
        
    def fetch_real_news(self, query, max_results=advanced_config.MAX_SEARCH_RESULTS):
        """Fetches related real news using DuckDuckGo search API."""
        try:
            results = self.ddgs.text(query, max_results=max_results)
            articles = []
            if results:
                for r in results:
                    articles.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", "") or r.get("href", ""),
                        "url": r.get("href", "")
                    })
                    if len(articles) >= max_results:
                        break
            return articles
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []
            
    def compute_similarity(self, input_text, articles):
        """
        Computes cosine similarity between input claim and fetched news snippets.
        Returns the maximum similarity found.
        """
        if not articles or not self.use_sbert:
            return 0.0
            
        input_embedding = self.sbert.encode(input_text, convert_to_tensor=True)
        
        max_sim = 0.0
        for article in articles:
            target_text = f"{article.get('title', '')}. {article.get('snippet', '')}"
            target_embedding = self.sbert.encode(target_text, convert_to_tensor=True)
            
            cos_score = util.cos_sim(input_embedding, target_embedding).item()
            if cos_score > max_sim:
                max_sim = cos_score
                
        return max_sim
        
    def validate(self, input_text):
        """Performs full validation: Search -> Embed -> Similarity."""
        query = self.extract_keywords(input_text)
        articles = self.fetch_real_news(query)
        sim_score = self.compute_similarity(input_text, articles)
        return sim_score, articles
