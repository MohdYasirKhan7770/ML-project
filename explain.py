from lime.lime_text import LimeTextExplainer
import re

class ExplainabilityEngine:
    def __init__(self, class_names=["Fake", "Real"]):
        # Binary classification
        self.explainer = LimeTextExplainer(class_names=class_names)
        
    def explain_prediction(self, text, predict_proba_fn, num_features=10):
        """
        Generates an explanation for a given prediction.
        predict_proba_fn takes a list of texts and returns a numpy 2D array [batch, probs]
        """
        clean_text = re.sub(r'\s+', ' ', text).strip()
        
        # Explainer requires minimum amount of text
        if len(clean_text.split()) < 3:
            return None
            
        exp = self.explainer.explain_instance(
            clean_text, 
            predict_proba_fn, 
            num_features=num_features
        )
        return exp
        
    def get_html(self, exp):
        """Returns the explanation as raw HTML for rendering."""
        if not exp:
            return ""
        return exp.as_html()
        
    def get_top_words(self, exp):
        """Returns the top words as a dictionary {word: weight}."""
        if not exp:
            return {}
        return dict(exp.as_list())
