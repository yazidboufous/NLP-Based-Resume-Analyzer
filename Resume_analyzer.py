import numpy as np
import spacy
from spacy.pipeline import EntityRuler
from spacy.matcher import PhraseMatcher
from collections import defaultdict
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Initialize PhraseMatcher with spaCy vocabulary
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

# Download NLTK data
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download(["stopwords", "wordnet"])


# Function to read file content
def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# Function to clean text data
def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"[^a-zA-Z\s]", "", text, re.I | re.A)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Function to preprocess text using spaCy
def preprocess_text(text, nlp):
    doc = nlp(text)
    tokens = [
        token.lemma_.lower().strip()
        for token in doc
        if not token.is_stop and not token.is_punct
    ]
    return tokens


# Create an entity ruler and load skill patterns from the specified file
ruler = nlp.add_pipe("entity_ruler")
skill_pattern_path = "jz_skill_patterns.jsonl"
ruler.from_disk(skill_pattern_path)


# Technical skill extraction function
def extract_technical_skills(input_text, nlp):
    # Preprocess the text
    filtered_tokens = preprocess_text(input_text, nlp)

    # Generate bigrams and trigrams
    bigrams_trigrams = list(map(" ".join, nltk.everygrams(filtered_tokens, 2, 3)))

    # Convert bigrams and trigrams into a spaCy Doc
    ngram_doc = nlp(" ".join(bigrams_trigrams))

    # Create a set for found skills
    found_skills = set()

    # Process the original text with the NLP pipeline to find skills
    for ent in nlp(input_text).ents:
        if ent.label_ == "SKILL":
            found_skills.add(ent.text)

    # Also check bigrams and trigrams
    for ent in ngram_doc.ents:
        if ent.label_ == "SKILL":
            found_skills.add(ent.text)

    return found_skills


# ----------------------------------------------------------------------------------------
def convert_file_to_2d_array(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    sections = content.split("\n\n")
    array_2d = [section.split("\n") for section in sections if section.strip()]
    return array_2d


soft_skills_data = convert_file_to_2d_array("soft_skills.txt")


soft_skills_vectors = {
    skill_group[0]: nlp(" ".join(skill_group)).vector
    for skill_group in soft_skills_data
}


# Soft skill extraction function
def extract_soft_skills(input_text, nlp, soft_skills_vectors):
    doc = nlp(input_text)
    matched_skills = defaultdict(int)

    # Consider only tokens that are likely to be relevant
    relevant_tokens = [
        token
        for token in doc
        if token.pos_ in ["NOUN", "ADJ", "VERB"] and token.has_vector
    ]

    # Compare with precomputed skill vectors
    for token in relevant_tokens:
        for skill, skill_vector in soft_skills_vectors.items():
            similarity = cosine_similarity(token.vector, skill_vector)
            if similarity > 0.6:
                matched_skills[skill] += 1

    sorted_skills = sorted(matched_skills.items(), key=lambda x: x[1], reverse=True)
    top_skills = [skill for skill, count in sorted_skills[:6]]

    return top_skills


# Define a cosine similarity function for efficiency
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Function to compare skills
def compare_skills(resume_skills, job_skills):
    resume_skill_set = set(resume_skills)
    job_skill_set = set(job_skills)

    common_skills = resume_skill_set.intersection(job_skill_set)
    missing_skills = job_skill_set.difference(resume_skill_set)
    return common_skills, missing_skills


# Read resume and job description
resume_text = read_file("resume.txt")
job_description_text = read_file("job_description.txt")


def calculate_success_score(common_skills, job_skills):
    if not job_skills:
        return 0  # Avoid division by zero
    score = len(common_skills) / len(job_skills) * 10
    return round(score, 2)  # Rounding to 2 decimal places


def process_texts(resume_text, job_description_text):
    # Extract technical skills
    resume_technical_skills = extract_technical_skills(resume_text, nlp)
    job_technical_skills = extract_technical_skills(job_description_text, nlp)

    # Extract soft skills
    resume_soft_skills = extract_soft_skills(resume_text, nlp, soft_skills_vectors)
    job_soft_skills = extract_soft_skills(
        job_description_text, nlp, soft_skills_vectors
    )

    # Compare skills
    common_technical_skills, missing_technical_skills = compare_skills(
        resume_technical_skills, job_technical_skills
    )
    common_soft_skills, missing_soft_skills = compare_skills(
        resume_soft_skills, job_soft_skills
    )

    technical_success_score = calculate_success_score(
        common_technical_skills, job_technical_skills
    )
    soft_success_score = calculate_success_score(common_soft_skills, job_soft_skills)
    overall_success_score = (technical_success_score + soft_success_score) / 2

    # Return results with success scores
    return {
        "common_technical_skills": common_technical_skills,
        "missing_technical_skills": missing_technical_skills,
        "common_soft_skills": common_soft_skills,
        "missing_soft_skills": missing_soft_skills,
        "technical_success_score": technical_success_score,
        "soft_success_score": soft_success_score,
        "overall_success_score": overall_success_score,
    }
