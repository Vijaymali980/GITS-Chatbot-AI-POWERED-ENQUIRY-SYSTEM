# 🎓 GITS AI-Powered College Enquiry Chatbot

An intelligent, interactive AI enquiry chatbot developed with **Streamlit** and **Natural Language Processing (NLP)** using **TF-IDF Vectorization** and **Cosine Similarity**. The system automatically answers common enquiries from prospective applicants, students, and parents about **admissions, courses, fees, scholarships, placements, hostel facilities**, and campus life.

---

## 🌟 Key Features

1. **Natural Language Processing (NLP) Engine**:
   - Uses **TF-IDF Vectorization** (Term Frequency-Inverse Document Frequency) with unigram and bigram ranges `(1, 2)`.
   - Employs **Cosine Similarity** to compute semantic relevance between user queries and the knowledge base.
   - Built-in **Synonym & Acronym Expansion** for college terminology (e.g., *CSE, ECE, B.Tech, MBA, tuition fees, hostels, packages*).
   - Conversational greeting & courtesy intent detection (*hello, hi, thanks, bye*).
   - Zero-dependency **Pure-Python TF-IDF fallback** ensuring 100% offline availability.

2. **Rich Interactive Web Interface (Streamlit)**:
   - Modern dark glassmorphic design system with glowing badges and responsive layout.
   - **Quick Question Chips** for instant one-click access to top FAQs.
   - **Form-based Query Input** with auto-clear and Enter-key submission.
   - **Relevance Confidence Meter** and category badges for each answer.
   - **Related Questions** recommendations to guide further enquiry.
   - **Fallback Responses** with intelligent suggestions and official college contact numbers when no high-confidence match is found.

3. **Session History & Analytics**:
   - Complete conversational history tracking during the session.
   - Live query metrics: total enquiries, success answer rate, fallback count, average confidence.
   - Category query distribution chart.
   - **Export Chat** button to download session logs as `.txt`.

4. **FAQ Explorer & Search**:
   - Searchable, category-filtered knowledge base of 45+ questions with expandable answers.
   - One-click "Ask this question" button for each FAQ.

---

## 📂 Project Architecture

```
├── app.py                 # Main Streamlit web application & NLP chatbot engine
├── college_data.json      # Structured college knowledge base (45 FAQs across 12 categories)
├── test_chatbot.py        # Automated test suite with 22 test questions
├── requirements.txt       # Python dependencies (Streamlit, Scikit-learn, Pandas, NLTK, Numpy)
├── Dockerfile             # Container configuration for Docker/Cloud Run/Render
├── Procfile               # Platform-as-a-Service deployment config (Heroku/Railway)
├── .streamlit/
│   └── config.toml        # Streamlit visual theme configuration
└── README.md              # Project documentation & deployment guide
```

---

## 🧠 NLP Pipeline Overview

```
User Query ("What is the fee for BTech computer science?")
                      │
                      ▼
 1. Query Normalization & Case Folding
                      │
                      ▼
 2. Greeting & Courtesy Intent Check (Hi, Thanks, Bye)
                      │
                      ▼
 3. Domain Synonym & Acronym Expansion ("fee" ➔ "fee fees tuition cost", "btech" ➔ "btech bachelor technology")
                      │
                      ▼
 4. TF-IDF Vectorization (Scikit-Learn / Pure-Python, N-grams 1-2)
                      │
                      ▼
 5. Cosine Similarity Calculation vs. Knowledge Base Matrix
                      │
                      ▼
 6. Threshold Evaluation
      ├── Score ≥ Threshold ➔ Display Best Match Answer + Confidence Meter + Related Questions
      └── Score < Threshold ➔ Trigger Fallback with Contact Info & "Did you mean..." Suggestion
```

---

## 📊 Dataset Structure (`college_data.json`)

The dataset contains official information for **Greenfield Institute of Technology & Science (GITS)**:
- **College Profile**: Established year, NAAC A+ accreditation, NIRF ranking, contact phone, admissions email, campus location.
- **FAQs (45 Questions across 12 Categories)**:
  - 📋 **Admissions**: Eligibility, JEE Main cutoffs, application procedure, required documents, lateral entry, NRI quota.
  - 📚 **Courses**: Undergraduate B.Tech specializations (CSE, AI/ML, ECE, ME, CE), Postgraduate M.Tech & MBA, certifications.
  - 💰 **Fees**: Annual tuition breakdown, semester payments, installment plans, application fee refund rules.
  - 🏆 **Scholarships**: Merit scholarships, EWS fee waivers, Sports quotas, Government NSP schemes.
  - 💼 **Placements**: 96% placement rate, ₹45 LPA highest package, ₹8.5 LPA average, 500+ recruiters (Google, Microsoft, Amazon).
  - 🏠 **Hostel**: AC & Non-AC room rates, mess food menu, safety, Wi-Fi, rules for staying off-campus.
  - 🏛️ **Infrastructure**: 150-acre campus, 40+ labs, central library (100k+ books), sports complex, Wi-Fi.
  - 👨‍🏫 **Faculty**: Ph.D. holders, student-to-faculty ratio (15:1), industry research collaborations.
  - 🎭 **Activities & Clubs**: Technical fests (TECHVITA), coding clubs, sports teams, internships.
  - 🚌 **Transport & Location**: 25 bus routes, airport/railway directions, nearest metro station.

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Chatbot
```bash
streamlit run app.py
```
The application will launch automatically in your browser at `http://localhost:8501`.

---

## 🧪 Testing the Chatbot

You can test the chatbot against various sample queries across all domains using the included test suite:

```bash
python test_chatbot.py
```

### Sample Test Questions to Try:
| Category | Query | Expected Behavior |
|----------|-------|-------------------|
| **Greeting** | *"Hello there"* | Friendly welcome message with quick topics |
| **Admissions** | *"How do I apply for B.Tech admission?"* | JEE score requirements, online application steps |
| **Courses** | *"Tell me about the Computer Science program"* | 4-year CSE intake, AI/ML tracks, 100% placement record |
| **Fees** | *"What is the annual fee for B.Tech?"* | Tuition fee breakdown (₹1.5L/year) & payment options |
| **Scholarships** | *"Are there scholarships for merit students?"* | Top 5% JEE score 50% tuition waiver details |
| **Placements** | *"What companies visit for placements?"* | Google, Microsoft, Amazon, Infosys, CTC stats |
| **Hostel** | *"What are the hostel room options and mess food?"* | Single/Double/Triple AC & Non-AC rates + multicuisine mess |
| **Fallback** | *"What is the weather on Mars?"* | Graceful fallback response with admissions contact info |

---

## 🌐 Online Deployment Guide

### Option 1: Streamlit Community Cloud (Recommended — Free & Instant)
1. Push this project repository to **GitHub**.
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **"New app"**.
4. Select your repository, branch (`main`), and set the main file path to `app.py`.
5. Click **"Deploy!"** — your chatbot will be live with a public URL in 1-2 minutes!

### Option 2: Hugging Face Spaces (Free)
1. Create a free account on [Hugging Face](https://huggingface.co).
2. Create a **New Space**, choose **Streamlit** as the Space SDK.
3. Upload `app.py`, `college_data.json`, `requirements.txt`, and `.streamlit/config.toml`.
4. The space will automatically build and provide a public hosted link.

### Option 3: Docker Deployment
Build and run the containerized application on any cloud provider (AWS, GCP Cloud Run, Azure, Render):
```bash
# Build the Docker image
docker build -t college-chatbot .

# Run the container
docker run -p 8501:8501 college-chatbot
```
Access at `http://localhost:8501`.

---

## 📄 License
This project is open-source and developed for educational and institutional enquiry automation.
