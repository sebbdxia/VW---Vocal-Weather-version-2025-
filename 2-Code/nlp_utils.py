# nlp_utils.py
import re
import spacy
import logging

nlp = spacy.load("fr_core_news_sm")

def spacy_analyze(text: str):
    doc = nlp(text)
    location = None
    forecast_days = None
    regex_match = re.search(r"(?:sur|pour)\s+(\d+)\s+jours", text, re.IGNORECASE)
    if regex_match:
        try:
            num = int(regex_match.group(1))
            if num in [3, 5, 7]:
                forecast_days = num
        except Exception as e:
            logging.error(f"Erreur extraction jours: {e}")
    if not forecast_days:
        for token in doc:
            if token.like_num:
                try:
                    num = int(token.text)
                    if num in [3, 5, 7]:
                        forecast_days = num
                        break
                except Exception:
                    continue
    for ent in doc.ents:
        if ent.label_ in ["LOC", "GPE"] and not location:
            location = ent.text
    if not location:
        location = "Paris"
    if not forecast_days:
        forecast_days = 7
    return location, forecast_days
