from flask import Flask, render_template, request
import re

from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory
)

from Sastrawi.Stemmer.StemmerFactory import (
    StemmerFactory
)

from transformers import AutoTokenizer


app = Flask(__name__)


# ==========================================
# SASTRAWI
# ==========================================

stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()


# ==========================================
# INDOBERT TOKENIZER
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    "indobenchmark/indobert-base-p1"
)


# ==========================================
# ROUTE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    text = ""
    case_folding = ""
    cleaning = ""
    stopword_removal = ""
    stemming = ""
    indobert_tokens = ""
    token_ids = ""


    if request.method == "POST":

        # 1. RAW TEXT
        text = request.form.get("text", "")


        # 2. CASE FOLDING
        case_folding = text.lower()


        # 3. CLEANING
        cleaning = re.sub(
            r"[^a-zA-Z0-9\s]",
            "",
            case_folding
        )


        # 4. STOPWORD REMOVAL
        stopword_removal = stopword_remover.remove(
            cleaning
        )


        # 5. STEMMING
        stemming = stemmer.stem(
            stopword_removal
        )


        # 6. INDOBERT TOKENIZATION
        tokens = tokenizer.tokenize(stemming)

        indobert_tokens = " | ".join(tokens)


        # 7. INDOBERT TOKEN IDS
        ids = tokenizer.convert_tokens_to_ids(tokens)

        token_ids = " | ".join(
            str(token_id)
            for token_id in ids
        )


        # DEBUGGING
        print("================================")
        print("RAW TEXT:", text)
        print("CASE FOLDING:", case_folding)
        print("CLEANING:", cleaning)
        print("STOPWORD REMOVAL:", stopword_removal)
        print("STEMMING:", stemming)
        print("INDOBERT TOKENS:", indobert_tokens)
        print("TOKEN IDS:", token_ids)
        print("================================")


    return render_template(
        "index.html",
        text=text,
        case_folding=case_folding,
        cleaning=cleaning,
        stopword_removal=stopword_removal,
        stemming=stemming,
        indobert_tokens=indobert_tokens,
        token_ids=token_ids
    )


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)