import streamlit as st
import joblib
import pandas as pd
import re
from urllib.parse import urlparse

st.set_page_config(page_title="AI Phishing URL Detector", page_icon="🛡️", layout="centered")

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

@st.cache_resource
def load_model():
    return joblib.load("live_model.pkl"), joblib.load("live_features.pkl")

model, feature_names = load_model()

# ---------- UI ----------
st.title("🛡️ AI-Based Phishing URL Detector")
st.markdown("Paste any URL below — the AI model analyzes **15 characteristics** and predicts if it's safe or a phishing trap.")

if "history" not in st.session_state:
    st.session_state.history = []

url = st.text_input("🔗 Enter a URL to scan:", placeholder="e.g. http://example-login.verify.com")

if st.button("🔍 Scan URL", type="primary"):
    if not url.strip():
        st.warning("Please enter a URL first.")
    else:
        feats = extract_url_features(url)
        X_in = pd.DataFrame([feats])[feature_names]
        pred = model.predict(X_in)[0]
        proba = model.predict_proba(X_in)[0]

        st.session_state.history.append("Phishing" if pred == 0 else "Legitimate")

        st.divider()
        if pred == 0:
            st.error(f"⚠️ PHISHING DETECTED — {proba[0]*100:.1f}% confidence")
            st.markdown("**Do not open this link or enter any personal information on it.**")
        else:
            st.success(f"✅ Looks Legitimate — {proba[1]*100:.1f}% confidence")

        with st.expander("🔬 What did the AI see? (extracted features)"):
            st.dataframe(pd.DataFrame([feats]).T.rename(columns={0: "Value"}))

# ---------- Sidebar stats ----------
st.sidebar.header("📊 Session Statistics")
if st.session_state.history:
    counts = pd.Series(st.session_state.history).value_counts()
    st.sidebar.bar_chart(counts)
    st.sidebar.write(f"Total scans this session: **{len(st.session_state.history)}**")
else:
    st.sidebar.write("No scans yet.")

st.sidebar.divider()
st.sidebar.markdown("**Model:** Random Forest (100 trees)")
st.sidebar.markdown("**Test accuracy:** 99.59%")
st.sidebar.markdown("**Dataset:** PhiUSIIL (235,795 URLs)")
