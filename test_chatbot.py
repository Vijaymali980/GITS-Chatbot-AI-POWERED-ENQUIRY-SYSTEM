"""
Test suite for GITS College Enquiry Chatbot NLP Engine.
Tests all required categories:
- Admissions
- Courses
- Fees
- Scholarships
- Placements
- Hostel Facilities
- Fallback handling
- Greetings & Courtesy
"""

import json
import re
from datetime import datetime

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


GREETING_PATTERNS = [
    r"^(hi|hello|hey|good\s*(morning|afternoon|evening)|namaste|greetings)(\s+.*)?$",
]
THANKS_PATTERNS = [
    r"^(thanks|thank\s*you|thx|much\s*appreciated|many\s*thanks)(\s+.*)?$",
]
BYE_PATTERNS = [
    r"^(bye|goodbye|see\s*you|cya)(\s+.*)?$",
]

SYNONYM_MAP = [
    (r"\bcse\b", "computer science engineering cse"),
    (r"\bece\b", "electronics communication engineering ece"),
    (r"\bmtech\b", "mtech master technology pg"),
    (r"\bbtech\b", "btech bachelor technology ug"),
    (r"\bmba\b", "mba master business administration"),
    (r"\bfees?\b", "fee fees tuition cost expenses"),
    (r"\bhostels?\b", "hostel accommodation mess rooms stay"),
    (r"\bplacements?\b", "placement placements salary packages recruiters companies"),
    (r"\bscholarships?\b", "scholarship scholarships financial aid waiver concession"),
    (r"\badmissions?\b", "admission admissions apply eligibility criteria procedure"),
]

def expand_synonyms(text: str) -> str:
    t = text.lower()
    for pattern, replacement in SYNONYM_MAP:
        t = re.sub(pattern, replacement, t)
    return t


class SklearnNLP:
    def __init__(self, documents):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b"
        )
        self.doc_matrix = self.vectorizer.fit_transform(documents)

    def get_similarities(self, query: str):
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.doc_matrix)[0]
        return scores


class TestChatbot:
    def __init__(self, data_path="college_data.json"):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.college_info = data["college_info"]
        self.faqs = data["faqs"]
        self.questions = [faq["question"] for faq in self.faqs]
        self.answers = [faq["answer"] for faq in self.faqs]
        self.categories = [faq["category"] for faq in self.faqs]

        # Enriched document list for training
        self.docs = [
            f"{faq['question']} {faq['category']} {faq['answer'][:100]}"
            for faq in self.faqs
        ]
        self.nlp = SklearnNLP(self.docs)

    def respond(self, query: str, threshold: float = 0.15):
        clean_q = query.strip()
        lower_q = clean_q.lower()

        # 1. Greetings check
        for pat in GREETING_PATTERNS:
            if re.match(pat, lower_q):
                return {
                    "type": "greeting",
                    "answer": f"Hello! 👋 Welcome to {self.college_info['name']} ({self.college_info['short_name']}). How can I assist you today? You can ask about admissions, courses, fees, scholarships, placements, hostel facilities, and more!",
                    "category": "greeting",
                    "confidence": 1.0,
                    "fallback": False
                }

        # 2. Thanks check
        for pat in THANKS_PATTERNS:
            if re.match(pat, lower_q):
                return {
                    "type": "thanks",
                    "answer": "You're very welcome! 😊 Feel free to ask if you have any more questions about GITS. We're always here to help!",
                    "category": "courtesy",
                    "confidence": 1.0,
                    "fallback": False
                }

        # 3. Bye check
        for pat in BYE_PATTERNS:
            if re.match(pat, lower_q):
                return {
                    "type": "bye",
                    "answer": "Goodbye! Have a great day ahead. Best wishes for your academic journey! 🎓",
                    "category": "courtesy",
                    "confidence": 1.0,
                    "fallback": False
                }

        # 4. Synonym expansion & TF-IDF similarity
        expanded_q = expand_synonyms(clean_q)
        scores = self.nlp.get_similarities(expanded_q)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        best_idx, best_score = ranked[0], float(scores[ranked[0]])

        if best_score < threshold:
            suggestion = self.questions[ranked[0]] if best_score > 0.05 else None
            return {
                "type": "fallback",
                "answer": "I couldn't find a specific answer for your query. Please contact our admissions desk at +91-80-2345-6789 or email admissions@gits.edu.in.",
                "category": "fallback",
                "confidence": best_score,
                "fallback": True,
                "suggestion": suggestion,
                "matched_question": None
            }

        related = [
            {"q": self.questions[i], "a": self.answers[i]}
            for i in ranked[1:3] if scores[i] >= threshold * 0.7
        ]

        return {
            "type": "answer",
            "answer": self.answers[best_idx],
            "category": self.categories[best_idx],
            "confidence": best_score,
            "fallback": False,
            "suggestion": None,
            "matched_question": self.questions[best_idx],
            "related": related
        }


def run_tests():
    print("=" * 60)
    print("GITS COLLEGE ENQUIRY CHATBOT TEST SUITE")
    print("=" * 60)

    bot = TestChatbot()

    test_queries = [
        # Greetings
        ("Hi there", "greeting"),
        ("Good morning", "greeting"),
        ("Thank you very much", "courtesy"),
        # Admissions
        ("How to apply for BTech admission?", "admissions"),
        ("What is the eligibility for B.Tech?", "admissions"),
        ("What documents are needed for admission?", "admissions"),
        # Courses
        ("What undergraduate courses are offered?", "courses"),
        ("Tell me about Computer Science Engineering", "courses"),
        ("Are there MBA programs?", "courses"),
        # Fees
        ("What is the fee structure for B.Tech?", "fees"),
        ("How much is the fee for MBA?", "fees"),
        ("What are the payment options?", "fees"),
        # Scholarships
        ("What scholarships are available for students?", "scholarships"),
        ("Is financial aid available for poor students?", "scholarships"),
        # Placements
        ("What is the highest placement package?", "placements"),
        ("Which companies visit for campus placement?", "placements"),
        ("What is the CSE placement record?", "placements"),
        # Hostel
        ("What are the hostel facilities and fees?", "hostel"),
        ("Is mess food good in the hostel?", "hostel"),
        ("Is hostel compulsory for all students?", "hostel"),
        # Fallback (out of domain)
        ("What is the weather on Mars?", "fallback"),
        ("Who won the 1994 world cup final?", "fallback"),
    ]

    passed = 0
    total = len(test_queries)

    for query, expected_cat in test_queries:
        res = bot.respond(query)
        cat = res["category"]
        conf = res["confidence"]
        matched_q = res.get("matched_question", "N/A")

        # Fallback check
        if expected_cat == "fallback":
            ok = res["fallback"] is True
        else:
            ok = (cat == expected_cat) and (not res["fallback"])

        status = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            passed += 1

        print(f"\nQuery: '{query}'")
        print(f"  Result: {status} | Cat: {cat} (Expected: {expected_cat}) | Conf: {conf:.2f}")
        if matched_q:
            print(f"  Matched: '{matched_q[:65]}...'")

    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} Passed ({passed/total*100:.1f}%)")
    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    run_tests()
