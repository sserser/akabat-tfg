# 📚 AKABAT – Advanced Knowledge Aggregator for Bibliometric Analysis and Trends
 
AKABAT is a novel analysis tool that allows researchers to quickly and effortlessly identify publication opportunities and generate statistical graphics to strengthen research sections. It helps researchers navigate their field by automatically identifying trends, tendencies and emerging sub-areas. AKABAT achieves this by using advanced natural language processing (NLP) techniques to analyze datasets of research papers and generate statistics to form a representation of the research landscape, identifying main categories and emerging sub-areas that have received limited research attention.

---

## ✨ Features

- 📂 Import bibliographic data (CSV, Scopus API)
- 🗂️ Advanced keyword filters (blacklist/whitelist)
- 🧠 Semantic clustering of keywords and authors (manual or automatic with Silhouette Score)
- 📊 Interactive visualizations (networks, maps, trends, bubble charts)
- 💾 Project saving and resume functionality

---

## ⚙️ Tech Stack

- Python 3.10+
- Django 4.x
- Pandas
- Plotly
- SentenceTransformers

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/sserser/akabat-tfg.git
cd akabat-tfg

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python manage.py runserver
Open http://127.0.0.1:8000/akabat in your browser.