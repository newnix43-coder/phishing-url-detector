import streamlit as st
import pandas as pd
import joblib
import re
from urllib.parse import urlparse

# ---------- Load model + feature order ----------
model = joblib.load('live_model.pkl')
feature_names = joblib.load('live_features.pkl')   # <-- guarantees correct order

# ---------- Feature extractor (identical to training) ----------
SUSPICIOUS_WORDS = ['login', 'verify', 'secure', 'account', 'update', 'free',
                    'bank', 'confirm', 'password', 'signin', 'paypal', 'wallet',
                    'bonus', 'lucky', 'click', 'urgent', 'suspend']

def extract_url_features(url):
    url = str(url).lower()
    parsed = urlparse(url if '://' in url else 'http://' + url)
    domain = parsed.netloc
    return {
        'url_length': len(url),
        'domain_length': len(domain),
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'num_underscores': url.count('_'),
        'num_slashes': url.count('/'),
        'num_digits': sum(c.isdigit() for c in url),
        'num_special': sum(url.count(c) for c in ['?', '=', '&', '%']),
        'has_at_symbol': int('@' in url),
        'has_ip_address': int(bool(re.match(r'\d+\.\d+\.\d+\.\d+', domain))),
        'is_https': int(parsed.scheme == 'https'),
        'num_subdomains': max(domain.count('.') - 1, 0),
        'digit_ratio': sum(c.isdigit() for c in url) / max(len(url), 1),
        'has_suspicious_word': int(any(w in url for w in SUSPICIOUS_WORDS)),
        'path_length': len(parsed.path),
    }

# ---------- UI ----------
st.set_page_config(page_title="Phishing URL Detector", page_icon="🛡️")
st.title("🛡️ Phishing URL Detector")
st.write("Enter a URL below to check whether it is **legitimate** or **phishing**.")

url = st.text_input("Enter URL:", placeholder="https://example.com")

if st.button("Check URL"):
    if not url.strip():
        st.warning("Please enter a URL first.")
    else:
        feats = extract_url_features(url)
        # THE FIX: build the row in the EXACT order the model was trained on
        X = pd.DataFrame([[feats[f] for f in feature_names]], columns=feature_names)

        pred = model.predict(X)[0]              # 0 = phishing, 1 = legitimate
        proba = model.predict_proba(X)[0]       # [P(phishing), P(legitimate)]

        if pred == 1:
            st.success(f"✅ **Legitimate** — confidence: {proba[1]*100:.1f}%")
        else:
            st.error(f"🚨 **Phishing detected!** — confidence: {proba[0]*100:.1f}%")
