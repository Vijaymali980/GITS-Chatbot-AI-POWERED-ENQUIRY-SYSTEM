import streamlit as st
import json
import numpy as np
import re
import math
from datetime import datetime
from collections import Counter

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ─────────────────────────────────────────────
#  Page Config  (MUST be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Mohini Institute of Technology and  Science AI College Enquiry Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');

:root {
    --bg-primary:    #080d14;
    --bg-secondary:  #0f1623;
    --bg-card:       #141d2e;
    --bg-card2:      #1a2540;
    --accent-1:      #00d4aa;
    --accent-2:      #7c5cfc;
    --accent-3:      #3b9eff;
    --accent-warn:   #ffa94d;
    --text-primary:  #e8f0fe;
    --text-muted:    #8899b0;
    --border:        rgba(255,255,255,0.07);
    --shadow-card:   0 8px 32px rgba(0,0,0,0.35);
    --radius:        18px;
    --radius-sm:     12px;
    --radius-xs:     8px;
    --transition:    all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,212,170,0.04) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(124,92,252,0.04) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080d14 0%, #0d1420 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

.main-header {
    background: linear-gradient(135deg, #0d1117 0%, #111827 40%, #0a1929 100%);
    border: 1px solid rgba(0,212,170,0.15);
    border-radius: var(--radius);
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-card);
}
.main-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle at 15% 50%, rgba(0,212,170,0.10) 0%, transparent 55%),
        radial-gradient(circle at 85% 20%, rgba(124,92,252,0.10) 0%, transparent 50%);
    animation: bgPulse 8s ease-in-out infinite alternate;
}
@keyframes bgPulse { from{opacity:0.7} to{opacity:1.0} }

.header-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4aa 0%, #7c5cfc 50%, #3b9eff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
    position: relative; z-index: 1;
}
.header-subtitle {
    color: var(--text-muted); font-size: 0.95rem;
    margin-bottom: 1rem; position: relative; z-index: 1;
}
.header-badges { display:flex; gap:0.6rem; flex-wrap:wrap; position:relative; z-index:1; }
.badge {
    background: rgba(0,212,170,0.10);
    border: 1px solid rgba(0,212,170,0.28);
    color: #00d4aa;
    padding: 0.22rem 0.85rem; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.02em;
}
.badge.purple { background:rgba(124,92,252,0.10); border-color:rgba(124,92,252,0.28); color:#a78bfa; }
.badge.blue   { background:rgba(59,158,255,0.10);  border-color:rgba(59,158,255,0.28);  color:#60a5fa; }
.badge.orange { background:rgba(255,169,77,0.10);  border-color:rgba(255,169,77,0.28);  color:#ffa94d; }

.chat-window {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.8rem 1.5rem;
    min-height: 480px;
    max-height: 560px;
    overflow-y: auto;
    box-shadow: inset 0 2px 20px rgba(0,0,0,0.3);
}
.chat-window::-webkit-scrollbar { width: 5px; }
.chat-window::-webkit-scrollbar-thumb { background: rgba(0,212,170,0.2); border-radius: 3px; }

.msg-row { display:flex; margin-bottom:1.4rem; animation: slideUp 0.3s ease-out; }
@keyframes slideUp { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
.msg-row.user { justify-content:flex-end; }
.msg-row.bot  { justify-content:flex-start; }

.msg-avatar {
    width:38px; height:38px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:1.15rem; flex-shrink:0;
    box-shadow:0 4px 12px rgba(0,0,0,0.3);
}
.msg-avatar.bot-av  { background:linear-gradient(135deg,#00d4aa,#0077cc); margin-right:0.75rem; }
.msg-avatar.user-av { background:linear-gradient(135deg,#7c5cfc,#cc5de8); margin-left:0.75rem; order:2; }

.msg-bubble { max-width:74%; padding:0.95rem 1.2rem; border-radius:20px; line-height:1.65; font-size:0.9rem; }
.msg-bubble.bot {
    background:var(--bg-card); border:1px solid var(--border);
    border-bottom-left-radius:5px; color:var(--text-primary); box-shadow:var(--shadow-card);
}
.msg-bubble.user {
    background:linear-gradient(135deg,#1a2d5a,#2d1a5e);
    border:1px solid rgba(124,92,252,0.22);
    border-bottom-right-radius:5px; color:#d4e8ff;
}
.msg-meta { font-size:0.7rem; color:var(--text-muted); margin-top:0.35rem; }
.msg-meta.bot  { text-align:left; }
.msg-meta.user { text-align:right; }

.cat-tag {
    display:inline-flex; align-items:center; gap:0.3rem;
    background:rgba(0,212,170,0.12); border:1px solid rgba(0,212,170,0.28);
    color:#00d4aa; font-size:0.68rem; padding:0.12rem 0.55rem;
    border-radius:999px; margin-bottom:0.55rem;
    font-weight:700; text-transform:uppercase; letter-spacing:0.06em;
}

.conf-wrap { margin-top:0.7rem; display:flex; align-items:center; gap:0.5rem; }
.conf-track { flex:1; height:3px; background:rgba(255,255,255,0.07); border-radius:2px; overflow:hidden; }
.conf-fill  { height:100%; border-radius:2px; transition:width 0.8s ease; }
.conf-label { font-size:0.68rem; color:var(--text-muted); white-space:nowrap; }

.fallback-bubble {
    background:rgba(255,169,77,0.07); border:1px solid rgba(255,169,77,0.22);
    border-radius:16px; padding:1rem 1.2rem; font-size:0.88rem;
    color:#ffd43b; line-height:1.7; max-width:74%;
}

.welcome-card {
    background:linear-gradient(135deg,rgba(0,212,170,0.05),rgba(124,92,252,0.05));
    border:1px dashed rgba(0,212,170,0.22);
    border-radius:var(--radius); padding:2rem; text-align:center; margin:0.5rem 0;
}
.welcome-emoji { font-size:3.5rem; margin-bottom:0.6rem; }
.welcome-title { font-family:'Outfit',sans-serif; font-size:1.25rem; font-weight:700; color:#00d4aa; margin-bottom:0.5rem; }
.welcome-text  { color:var(--text-muted); font-size:0.88rem; line-height:1.7; }
.welcome-topics { display:flex; flex-wrap:wrap; justify-content:center; gap:0.4rem; margin-top:1rem; }
.topic-chip {
    background:rgba(59,158,255,0.1); border:1px solid rgba(59,158,255,0.22);
    color:#60a5fa; padding:0.2rem 0.7rem; border-radius:999px; font-size:0.75rem; font-weight:500;
}

.stTextInput > div > div > input {
    background: var(--bg-card) !important; border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important; color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important; padding: 0.75rem 1.1rem !important;
    font-size: 0.92rem !important; transition: var(--transition) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent-1) !important;
    box-shadow: 0 0 0 3px rgba(0,212,170,0.10) !important;
    background: var(--bg-card2) !important;
}
.stTextInput > div > div > input::placeholder { color: var(--text-muted) !important; }

.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #0099ff) !important;
    color: #060c14 !important; border: none !important;
    border-radius: var(--radius-xs) !important; font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important; padding: 0.6rem 1.4rem !important;
    font-size: 0.88rem !important; transition: var(--transition) !important;
    box-shadow: 0 4px 15px rgba(0,212,170,0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,212,170,0.35) !important;
    filter: brightness(1.08) !important;
}

.stat-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 1rem 1.1rem;
    margin-bottom: 0.7rem; transition: var(--transition);
}
.stat-card:hover { border-color:rgba(0,212,170,0.35); background:var(--bg-card2); transform:translateX(3px); }
.stat-label { font-size:0.72rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.2rem; }
.stat-value { font-size:1.3rem; font-weight:700; color:#00d4aa; }

.section-label {
    font-size:0.72rem; text-transform:uppercase; letter-spacing:0.09em;
    color:var(--text-muted); margin-bottom:0.6rem; margin-top:0.8rem;
    display:flex; align-items:center; gap:0.4rem;
}
.section-label::after { content:''; flex:1; height:1px; background:var(--border); }

.info-card {
    background:linear-gradient(135deg,rgba(0,212,170,0.06),rgba(59,158,255,0.04));
    border:1px solid rgba(0,212,170,0.18); border-radius:var(--radius-sm);
    padding:1rem 1.1rem; font-size:0.82rem; line-height:1.75; margin-bottom:0.7rem;
}
.info-card-title { font-weight:700; color:#00d4aa; margin-bottom:0.4rem; font-size:0.88rem; }

.stSelectbox > div > div {
    background: var(--bg-card) !important; border-color: var(--border) !important;
    color: var(--text-primary) !important; border-radius: var(--radius-xs) !important;
}

[data-testid="stMetric"] {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 0.8rem 1rem;
}
[data-testid="stMetricValue"] { color: #00d4aa !important; font-family:'Outfit',sans-serif !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Pure-Python NLP Engine (Zero network deps)
# ─────────────────────────────────────────────
STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do","does",
    "did","will","would","shall","should","may","might","must","can","could",
    "i","me","my","we","our","you","your","he","him","his","she","her","it",
    "its","they","them","their","what","which","who","this","that","these",
    "those","am","if","then","when","there","how","all","both","each","some",
    "such","no","not","only","same","so","than","too","very","just","about",
    "tell","please","need","want","like","know","get","also","many","any",
    "much","up","out","now","here","from","s","t","don",
}

def _stem(word):
    suffixes = [
        ("ational","ate"),("tional","tion"),("izations","ize"),("ization","ize"),
        ("ising","ise"),("izing","ize"),("ations","ate"),("nesses",""),
        ("iveness","ive"),("fulness","ful"),("ousness","ous"),("alists","al"),
        ("alism","al"),("aliti","al"),("iviti","ive"),("biliti","ble"),
        ("ings",""),("edly",""),("ingly",""),("ness",""),("ments",""),("ment",""),
        ("ful",""),("less",""),("ous",""),("ive",""),("ize",""),("ise",""),
        ("ate",""),("ical",""),("ics",""),("ies","y"),("ied","y"),
        ("ers",""),("ing",""),("est",""),("ed",""),("er",""),("es",""),("s",""),
    ]
    w = word.lower()
    if len(w) <= 3:
        return w
    for suffix, rep in suffixes:
        if w.endswith(suffix) and len(w) - len(suffix) >= 2:
            return w[:len(w)-len(suffix)] + rep
    return w


# ─────────────────────────────────────────────
#  Conversational Intent Patterns & Synonyms
# ─────────────────────────────────────────────
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
    (r"\bmtech\b", "mtech master technology postgraduate"),
    (r"\bbtech\b", "btech bachelor technology undergraduate"),
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


class SklearnTFIDFEngine:
    """Production TF-IDF + Cosine Similarity NLP Engine using scikit-learn."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b"
        )
        self.doc_matrix = None

    def fit(self, documents):
        self.doc_matrix = self.vectorizer.fit_transform(documents)

    def similarities(self, query):
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.doc_matrix)[0]
        return scores.tolist()


class TFIDFEngine:
    """Pure-Python TF-IDF Engine fallback (runs with zero dependencies)."""
    def __init__(self):
        self.vocab = {}
        self.idf = []
        self.doc_vecs = []

    @staticmethod
    def preprocess(text):
        text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        return [_stem(t) for t in text.split() if t not in STOPWORDS and len(t) > 1]

    def fit(self, documents):
        tokenised = [self.preprocess(d) for d in documents]
        N = len(tokenised)
        word_set = set(t for tok in tokenised for t in tok)
        self.vocab = {w: i for i, w in enumerate(sorted(word_set))}
        V = len(self.vocab)
        df = [0] * V
        for tok in tokenised:
            for t in set(tok):
                if t in self.vocab:
                    df[self.vocab[t]] += 1
        self.idf = [math.log((N + 1) / (d + 1)) + 1 for d in df]
        self.doc_vecs = [self._vec(tok) for tok in tokenised]

    def _vec(self, tokens):
        V = len(self.vocab)
        vec = [0.0] * V
        cnt = Counter(tokens)
        total = sum(cnt.values()) or 1
        for t, c in cnt.items():
            if t in self.vocab:
                idx = self.vocab[t]
                vec[idx] = (c / total) * self.idf[idx]
        norm = math.sqrt(sum(x*x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def transform(self, text):
        return self._vec(self.preprocess(text))

    def similarities(self, qv):
        return [sum(a*b for a,b in zip(qv, dv)) for dv in self.doc_vecs]


class CollegeChatbot:
    def __init__(self, data_path="college_data.json"):
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.college_info = raw["college_info"]
        self.faqs = raw["faqs"]
        self.questions  = [faq["question"]  for faq in self.faqs]
        self.answers    = [faq["answer"]    for faq in self.faqs]
        self.categories = [faq["category"]  for faq in self.faqs]

        # Enriched document representations for higher matching recall
        self.enriched_docs = [
            f"{faq['question']} {faq['category']} {faq['answer'][:120]}"
            for faq in self.faqs
        ]

        if SKLEARN_AVAILABLE:
            self.engine = SklearnTFIDFEngine()
            self.engine.fit(self.enriched_docs)
            self.engine_name = "Scikit-Learn TF-IDF + Cosine Similarity (1-2 N-grams)"
        else:
            self.engine = TFIDFEngine()
            self.engine.fit(self.questions)
            self.engine_name = "Pure-Python TF-IDF Engine"

    def get_response(self, query, threshold=0.15):
        clean_q = query.strip()
        lower_q = clean_q.lower()

        # 1. Polite greetings handling
        for pat in GREETING_PATTERNS:
            if re.match(pat, lower_q):
                return {
                    "answer": f"Hello! 👋 Welcome to {self.college_info['name']} ({self.college_info['short_name']}). How can I assist you today? You can ask me about admissions, courses, fees, scholarships, placements, hostel facilities, campus life, and more!",
                    "question": "Greeting",
                    "category": "general",
                    "confidence": 1.0,
                    "fallback": False,
                    "greeting": True,
                    "suggestion": None,
                    "related": []
                }

        # 2. Courtesy / Thanks handling
        for pat in THANKS_PATTERNS:
            if re.match(pat, lower_q):
                return {
                    "answer": "You're very welcome! 😊 Feel free to ask if you have any more questions about GITS. We're here to help you anytime!",
                    "question": "Courtesy",
                    "category": "general",
                    "confidence": 1.0,
                    "fallback": False,
                    "greeting": True,
                    "suggestion": None,
                    "related": []
                }

        # 3. Courtesy / Goodbye handling
        for pat in BYE_PATTERNS:
            if re.match(pat, lower_q):
                return {
                    "answer": "Goodbye! Have a great day ahead and best wishes for your academic journey! 🎓",
                    "question": "Courtesy",
                    "category": "general",
                    "confidence": 1.0,
                    "fallback": False,
                    "greeting": True,
                    "suggestion": None,
                    "related": []
                }

        # 4. Synonym expansion & TF-IDF similarity computation
        expanded_q = expand_synonyms(clean_q)
        if SKLEARN_AVAILABLE:
            scores = self.engine.similarities(expanded_q)
        else:
            qv = self.engine.transform(expanded_q)
            scores = self.engine.similarities(qv)

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        best_idx, best_score = ranked[0], float(scores[ranked[0]])
        suggestion = self.questions[ranked[0]] if (len(ranked) > 0 and best_score > 0.05) else None

        cutoff = threshold if SKLEARN_AVAILABLE else 0.12
        if best_score < cutoff:
            return {
                "answer": None,
                "question": None,
                "category": None,
                "confidence": best_score,
                "fallback": True,
                "greeting": False,
                "suggestion": suggestion,
                "related": []
            }

        related = [
            {"q": self.questions[i], "a": self.answers[i]}
            for i in ranked[1:3] if scores[i] >= cutoff * 0.65
        ]

        return {
            "answer": self.answers[best_idx],
            "question": self.questions[best_idx],
            "category": self.categories[best_idx],
            "confidence": best_score,
            "fallback": False,
            "greeting": False,
            "suggestion": None,
            "related": related
        }

    def get_categories(self):
        return sorted(set(self.categories))

    def get_faqs_by_category(self, cat):
        return [(f["question"], f["answer"]) for f in self.faqs if f["category"] == cat]

    def search_faqs(self, keyword):
        kw = keyword.lower()
        return [f for f in self.faqs if kw in f["question"].lower() or kw in f["answer"].lower()]


@st.cache_resource
def load_chatbot():
    return CollegeChatbot("college_data.json")


def init_state():
    defaults = {
        "messages":[],"total_queries":0,"answered":0,"fallbacks":0,
        "start_time":datetime.now().strftime("%H:%M"),"category_counts":{},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
chatbot = load_chatbot()
college_info = chatbot.college_info


def cat_icon(cat):
    return {"admissions":"📋","courses":"📚","fees":"💰","scholarships":"🏆",
            "placements":"💼","hostel":"🏠","infrastructure":"🏛️","faculty":"👨‍🏫",
            "activities":"🎭","research":"🔬","transport":"🚌","general":"ℹ️"}.get(cat,"💬")

def conf_color(s):
    return "#00d4aa" if s>=0.55 else ("#3b9eff" if s>=0.30 else "#ffa94d")


def render_message(msg, idx):
    role = msg["role"]
    if role == "user":
        st.markdown(f"""
        <div class="msg-row user">
            <div>
                <div class="msg-bubble user">{msg['content']}</div>
                <div class="msg-meta user">{msg.get('time','')}</div>
            </div>
            <div class="msg-avatar user-av">👤</div>
        </div>""", unsafe_allow_html=True)
    else:
        data = msg.get("data", {})
        if data.get("fallback"):
            sugg = data.get("suggestion","")
            sugg_html = f'<br>💡 <em>Did you mean: "{sugg[:80]}..."?</em>' if sugg else ""
            st.markdown(f"""
            <div class="msg-row bot">
                <div class="msg-avatar bot-av">🎓</div>
                <div>
                    <div class="fallback-bubble">
                        ⚠️ I couldn't find a specific answer for that query.{sugg_html}<br><br>
                        <strong>Try asking about:</strong> admissions · courses · fees · scholarships · placements · hostel · transport<br><br>
                        📞 <strong>+91-80-2345-6789</strong> &nbsp;|&nbsp; 📧 <strong>admissions@gits.edu.in</strong>
                    </div>
                    <div class="msg-meta bot">{msg.get('time','')}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        elif data.get("greeting"):
            icon  = "👋"
            answer = data.get("answer","").replace("\n","<br>")
            st.markdown(f"""
            <div class="msg-row bot">
                <div class="msg-avatar bot-av">🎓</div>
                <div style="max-width:74%">
                    <div class="msg-bubble bot">
                        <div class="cat-tag">{icon} GREETING</div>
                        <div>{answer}</div>
                    </div>
                    <div class="msg-meta bot">{msg.get('time','')}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            cat   = data.get("category","general")
            icon  = cat_icon(cat)
            score = data.get("confidence",0)
            color = conf_color(score)
            pct   = min(int(score * 140), 100)
            answer = data.get("answer","").replace("\n","<br>")
            related = data.get("related",[])
            rel_html = ""
            if related:
                items = "".join(f'<li style="margin-bottom:0.3rem">{r["q"]}</li>' for r in related[:2])
                rel_html = f"""
                <div style="margin-top:0.8rem;padding-top:0.7rem;border-top:1px solid rgba(255,255,255,0.07)">
                    <div style="font-size:0.72rem;color:#6b7280;margin-bottom:0.3rem">🔗 Related questions:</div>
                    <ul style="font-size:0.78rem;color:#6b7280;padding-left:1.2rem;margin:0">{items}</ul>
                </div>"""
            st.markdown(f"""
            <div class="msg-row bot">
                <div class="msg-avatar bot-av">🎓</div>
                <div style="max-width:74%">
                    <div class="msg-bubble bot">
                        <div class="cat-tag">{icon} {cat.upper()}</div>
                        <div>{answer}</div>
                        {rel_html}
                        <div class="conf-wrap">
                            <div class="conf-track">
                                <div class="conf-fill" style="width:{pct}%;background:{color}"></div>
                            </div>
                            <span class="conf-label">Relevance {pct}%</span>
                        </div>
                    </div>
                    <div class="msg-meta bot">{msg.get('time','')}</div>
                </div>
            </div>""", unsafe_allow_html=True)


def process_query(query):
    query = query.strip()
    if not query:
        return
    now = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role":"user","content":query,"time":now})
    st.session_state.total_queries += 1
    result = chatbot.get_response(query)
    if result["fallback"]:
        st.session_state.fallbacks += 1
    else:
        st.session_state.answered += 1
        cat = result.get("category","general")
        cc  = st.session_state.category_counts
        cc[cat] = cc.get(cat,0) + 1
    st.session_state.messages.append({"role":"bot","content":"","data":result,"time":now})


# ═══════════════════ SIDEBAR ═══════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:1.2rem 0 0.6rem">
        <div style="font-size:3rem;filter:drop-shadow(0 0 12px rgba(0,212,170,0.4))">🎓</div>
        <div style="font-family:'Outfit',sans-serif;font-size:1.2rem;font-weight:800;
                    background:linear-gradient(135deg,#00d4aa,#7c5cfc);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            {college_info['short_name']} Chatbot
        </div>
        <div style="font-size:0.72rem;color:#6b7280;margin-top:0.2rem;letter-spacing:0.05em">
            AI-POWERED ENQUIRY SYSTEM
        </div>
    </div>
    <hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);margin:0.6rem 0">
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">📊 Session Stats</div>', unsafe_allow_html=True)
    total   = st.session_state.total_queries
    answered= st.session_state.answered
    rate    = int(answered/total*100) if total else 0

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Queries</div>'
                    f'<div class="stat-value">{total}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-label">Answered</div>'
                    f'<div class="stat-value" style="color:#7c5cfc">{rate}%</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stat-card"><div class="stat-label">Session started</div>'
                f'<div class="stat-value" style="font-size:1rem;color:#3b9eff">{st.session_state.start_time}</div></div>',
                unsafe_allow_html=True)

    cc = st.session_state.category_counts
    if cc:
        top_cat = max(cc, key=cc.get)
        st.markdown(f'<div class="stat-card"><div class="stat-label">Top Category</div>'
                    f'<div class="stat-value" style="font-size:0.95rem;color:#ffa94d">'
                    f'{cat_icon(top_cat)} {top_cat.capitalize()}</div></div>', unsafe_allow_html=True)

    st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);margin:0.6rem 0">',
                unsafe_allow_html=True)

    st.markdown('<div class="section-label">📂 Browse by Category</div>', unsafe_allow_html=True)
    categories   = chatbot.get_categories()
    cat_options  = ["── Choose a topic ──"] + [f"{cat_icon(c)} {c.capitalize()}" for c in categories]
    selected_cat = st.selectbox("Topic", cat_options, label_visibility="collapsed", key="cat_select")

    if selected_cat != "── Choose a topic ──":
        cat_key     = selected_cat.split(" ", 1)[1].lower()
        faqs_in_cat = chatbot.get_faqs_by_category(cat_key)
        st.markdown(f'<div style="font-size:0.75rem;color:#6b7280;margin-bottom:0.4rem">'
                    f'{len(faqs_in_cat)} questions in this topic</div>', unsafe_allow_html=True)
        for i, (q, _) in enumerate(faqs_in_cat[:6]):
            label = f"❓ {q[:45]}…" if len(q) > 45 else f"❓ {q}"
            if st.button(label, key=f"catq_{cat_key}_{i}", use_container_width=True):
                process_query(q)
                st.rerun()

    st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);margin:0.6rem 0">',
                unsafe_allow_html=True)

    info = chatbot.college_info
    st.markdown(f"""
    <div class="info-card">
        <div class="info-card-title">🏛️ {info['short_name']}</div>
        <div>📍 {info['location']}</div>
        <div>📞 {info['phone']}</div>
        <div>🌐 {info['website']}</div>
        <div style="margin-top:0.5rem;color:#00d4aa;font-size:0.75rem">{info['accreditation']}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.07);margin:0.6rem 0">',
                unsafe_allow_html=True)

    if st.button("🗑️  Clear Chat History", use_container_width=True):
        st.session_state.messages        = []
        st.session_state.total_queries   = 0
        st.session_state.answered        = 0
        st.session_state.fallbacks       = 0
        st.session_state.category_counts = {}
        st.rerun()

    if st.session_state.messages:
        chat_text = "\n".join(
            f"[{m['time']}] {'YOU' if m['role']=='user' else 'BOT'}: "
            + (m['content'] if m['role']=='user'
               else (m['data'].get('answer','') or '[Fallback response]'))
            for m in st.session_state.messages
        )
        st.download_button(
            "⬇️  Export Chat (.txt)", data=chat_text,
            file_name=f"gits_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain", use_container_width=True,
        )


# ═══════════════════ MAIN ═══════════════════
st.markdown(f"""
<div class="main-header">
    <p class="header-title">🎓 {college_info['short_name']} AI Enquiry Chatbot</p>
    <p class="header-subtitle">
        {college_info['name']} &nbsp;·&nbsp;
        TF-IDF + Cosine Similarity NLP &nbsp;·&nbsp; ⚡ 100% Offline
    </p>
    <div class="header-badges">
        <span class="badge">✅ NAAC A+</span>
        <span class="badge purple">🏆 NIRF Rank 47</span>
        <span class="badge blue">🤖 AI-Powered</span>
        <span class="badge orange">⚡ Offline NLP</span>
        <span class="badge">{len(chatbot.faqs)} FAQs</span>
        <span class="badge purple">24/7 Available</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab_chat, tab_faq, tab_analytics, tab_about = st.tabs([
    "💬  Chat", "📋  FAQ Browser", "📊  Analytics", "ℹ️  About"
])


# ── TAB 1: CHAT ──────────────────────────────
with tab_chat:
    col_main, col_right = st.columns([3, 1])
    with col_main:
        st.markdown('<div class="section-label">⚡ Quick Questions</div>', unsafe_allow_html=True)
        quick_qs = [
            ("💰","B.Tech fee structure?"),("💼","Placement statistics?"),
            ("📚","Courses offered?"),("🏠","Hostel facilities?"),
            ("🏆","Scholarships available?"),("📋","Admission process?"),
        ]
        qcols = st.columns(len(quick_qs))
        for ci, (col, (icon, q)) in enumerate(zip(qcols, quick_qs)):
            with col:
                label = f"{icon} {q[:16]}…" if len(q) > 16 else f"{icon} {q}"
                if st.button(label, key=f"quick_{ci}", use_container_width=True):
                    process_query(q)
                    st.rerun()

        st.markdown('<div class="section-label" style="margin-top:1rem">💬 Conversation</div>',
                    unsafe_allow_html=True)
        with st.container():
            if not st.session_state.messages:
                st.markdown(f"""
                <div class="welcome-card">
                    <div class="welcome-emoji">👋</div>
                    <div class="welcome-title">Hello! I'm the {college_info['short_name']} Virtual Assistant</div>
                    <div class="welcome-text">
                        Ask me anything about <strong>admissions</strong>, <strong>courses</strong>,
                        <strong>fees</strong>, <strong>scholarships</strong>, <strong>placements</strong>,
                        <strong>hostel</strong>, <strong>infrastructure</strong>, and more.
                    </div>
                    <div class="welcome-topics">
                        <span class="topic-chip">📋 Admissions</span>
                        <span class="topic-chip">📚 Courses</span>
                        <span class="topic-chip">💰 Fees</span>
                        <span class="topic-chip">🏆 Scholarships</span>
                        <span class="topic-chip">💼 Placements</span>
                        <span class="topic-chip">🏠 Hostel</span>
                        <span class="topic-chip">🏛️ Infrastructure</span>
                        <span class="topic-chip">🚌 Transport</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for idx, msg in enumerate(st.session_state.messages):
                    render_message(msg, idx)

        st.markdown('<div class="section-label" style="margin-top:1rem">✍️ Ask a Question</div>',
                    unsafe_allow_html=True)
        with st.form(key="chat_query_form", clear_on_submit=True):
            inp_col, btn_col = st.columns([5, 1])
            with inp_col:
                user_input = st.text_input(
                    "Query", placeholder="Ask about admissions, courses, fees, scholarships, placements, hostel…",
                    label_visibility="collapsed", key="chat_input",
                )
            with btn_col:
                send_clicked = st.form_submit_button("Send 🚀", use_container_width=True)

        if send_clicked and user_input:
            process_query(user_input)
            st.rerun()

    with col_right:
        st.markdown('<div class="section-label">🏆 Key Highlights</div>', unsafe_allow_html=True)
        highlights = [
            ("96%","Placement Rate","#00d4aa"),("₹8.5L","Avg Package","#7c5cfc"),
            ("₹45L","Highest CTC","#3b9eff"),("500+","Recruiters","#ffa94d"),
            ("NAAC A+","Accreditation","#00d4aa"),("150 Ac","Campus","#3b9eff"),
            ("280+","Faculty","#7c5cfc"),("10K+","Students","#ffa94d"),
        ]
        for val, label, color in highlights:
            st.markdown(f'<div class="stat-card"><div class="stat-label">{label}</div>'
                        f'<div class="stat-value" style="font-size:1.05rem;color:{color}">{val}</div></div>',
                        unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.76rem;color:#6b7280;line-height:1.8;margin-top:0.5rem">
            <div style="font-weight:600;color:#8899b0;margin-bottom:0.4rem">🤖 NLP Pipeline</div>
            <div>1️⃣ Tokenise + stem query</div>
            <div>2️⃣ TF-IDF vectorisation</div>
            <div>3️⃣ Cosine similarity</div>
            <div>4️⃣ Return best match</div>
        </div>""", unsafe_allow_html=True)


# ── TAB 2: FAQ BROWSER ───────────────────────
with tab_faq:
    st.markdown("### 📋 Browse All FAQs")
    sc, cc_ = st.columns([3,1])
    with sc:
        search_kw = st.text_input("🔍 Search FAQs", placeholder="Search by keyword…", key="faq_search")
    with cc_:
        faq_cat_filter = st.selectbox(
            "Filter", ["All Categories"] + [f"{cat_icon(c)} {c.capitalize()}"
                                            for c in chatbot.get_categories()],
            key="faq_cat_filter"
        )
    all_faqs = chatbot.faqs
    if search_kw:
        all_faqs = chatbot.search_faqs(search_kw)
    if faq_cat_filter != "All Categories":
        ck = faq_cat_filter.split(" ",1)[1].lower()
        all_faqs = [f for f in all_faqs if f["category"] == ck]

    st.markdown(f'<div style="color:#6b7280;font-size:0.8rem;margin-bottom:0.8rem">'
                f'Showing {len(all_faqs)} of {len(chatbot.faqs)} FAQs</div>', unsafe_allow_html=True)

    for faq in all_faqs:
        icon = cat_icon(faq["category"])
        with st.expander(f'{icon} {faq["question"]}'):
            st.markdown(f"""
            <div style="color:#e8f0fe;font-size:0.88rem;line-height:1.7;padding:0.5rem 0">
                {faq['answer']}
            </div>
            <div style="margin-top:0.6rem">
                <span class="badge" style="font-size:0.7rem">{icon} {faq['category'].capitalize()}</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Ask this question", key=f"faq_ask_{faq['id']}"):
                process_query(faq["question"])
                st.rerun()


# ── TAB 3: ANALYTICS ─────────────────────────
with tab_analytics:
    st.markdown("### 📊 Session Analytics")
    total_q  = st.session_state.total_queries
    ans_q    = st.session_state.answered
    fall_q   = st.session_state.fallbacks
    rate_q   = int(ans_q/total_q*100) if total_q else 0

    bot_msgs = [m for m in st.session_state.messages if m["role"]=="bot"]
    avg_conf = 0.0
    if bot_msgs:
        confs = [m["data"].get("confidence",0) for m in bot_msgs]
        avg_conf = sum(confs)/len(confs)

    m1,m2,m3,m4 = st.columns(4)
    with m1: st.metric("Total Queries", total_q)
    with m2: st.metric("Answered", ans_q, delta=f"{rate_q}% rate")
    with m3: st.metric("Fallbacks", fall_q)
    with m4: st.metric("Avg Confidence", f"{min(int(avg_conf*140),100)}%")

    cc2 = st.session_state.category_counts
    if cc2:
        st.markdown('<div class="section-label" style="margin-top:1rem">📂 Queries by Category</div>',
                    unsafe_allow_html=True)
        max_val = max(cc2.values())
        for cat, count in sorted(cc2.items(), key=lambda x: x[1], reverse=True):
            pct2 = int(count/max_val*100)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.6rem">
                <div style="width:130px;font-size:0.82rem;color:#8899b0">{cat_icon(cat)} {cat.capitalize()}</div>
                <div style="flex:1;height:8px;background:rgba(255,255,255,0.07);border-radius:4px;overflow:hidden">
                    <div style="width:{pct2}%;height:100%;background:linear-gradient(90deg,#00d4aa,#7c5cfc);border-radius:4px"></div>
                </div>
                <div style="width:30px;font-size:0.82rem;color:#00d4aa;font-weight:700">{count}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Start a conversation to see analytics here.")

    if st.session_state.messages:
        st.markdown('<div class="section-label" style="margin-top:1.2rem">📜 Conversation History</div>',
                    unsafe_allow_html=True)
        import pandas as pd
        rows = []
        for m in st.session_state.messages:
            if m["role"] == "user":
                rows.append({"Time":m.get("time",""),"Role":"👤 User",
                             "Message":m["content"],"Category":"—","Confidence":"—"})
            else:
                d = m.get("data",{})
                rows.append({"Time":m.get("time",""),"Role":"🎓 Bot",
                             "Message":(d.get("answer","") or "Fallback response")[:80]+"…",
                             "Category":d.get("category","fallback") or "fallback",
                             "Confidence":f"{min(int(d.get('confidence',0)*140),100)}%" if not d.get("fallback") else "N/A"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── TAB 4: ABOUT ─────────────────────────────
with tab_about:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        ### 🎓 About This Chatbot
        The **{college_info['name']} ({college_info['short_name']}) AI College Enquiry Chatbot** is a fully offline, NLP-powered virtual assistant
        designed to automatically answer common questions from students, parents, and applicants.

        ---
        #### 🔧 Technology Stack
        | Component | Technology |
        |-----------|-----------|
        | Framework | Streamlit (Python) |
        | Active NLP Engine | **{chatbot.engine_name}** |
        | Similarity Metric | Cosine Similarity |
        | Vectorizer | TF-IDF (1-2 N-grams, Sublinear TF) |
        | Preprocessing | Stopwords removal, synonym expansion, case folding |
        | Dataset | JSON Knowledge Base (45 FAQs) |
        | Network Required | ❌ None — 100% Offline |

        ---
        #### 🧠 NLP Pipeline
        1. **Query Preprocessing** – Case folding, cleaning, conversational intent detection
        2. **Domain Synonym Expansion** – Expanding abbreviations (CSE, ECE, fees, hostel)
        3. **TF-IDF Vectorisation** – Computing unigram + bigram TF-IDF vectors
        4. **Cosine Similarity** – Vector dot product matching against FAQ corpus
        5. **Confidence & Fallback** – If score ≥ threshold → return answer + related topics; else → fallback with intelligent suggestions
        """)
    with col_b:
        st.markdown("""
        ### 📚 Knowledge Base Coverage

        | Category | FAQs |
        |----------|------|
        | 📋 Admissions | 7 |
        | 📚 Courses | 4 |
        | 💰 Fees | 3 |
        | 🏆 Scholarships | 4 |
        | 💼 Placements | 4 |
        | 🏠 Hostel | 4 |
        | 🏛️ Infrastructure | 3 |
        | 👨‍🏫 Faculty | 2 |
        | 🎭 Activities | 3 |
        | 🔬 Research | 1 |
        | 🚌 Transport | 2 |
        | ℹ️ General | 4 |

        ---
        #### 🚀 Deployment Options
        - **Local**: `streamlit run app.py`
        - **Streamlit Cloud**: Push to GitHub → share.streamlit.io
        - **Heroku/Railway**: `Procfile` → `web: streamlit run app.py --server.port $PORT`
        - **Docker**: `EXPOSE 8501` + `CMD streamlit run app.py`
        """)
        info = chatbot.college_info
        st.markdown(f"""
        <div class="info-card" style="margin-top:1rem">
            <div class="info-card-title">🏛️ {info['name']}</div>
            <div>📍 {info['location']}</div>
            <div>📞 {info['phone']}</div>
            <div>📧 {info['email']}</div>
            <div>🌐 {info['website']}</div>
            <div style="margin-top:0.5rem;color:#00d4aa;font-size:0.75rem">{info['accreditation']}</div>
        </div>
        """, unsafe_allow_html=True)
