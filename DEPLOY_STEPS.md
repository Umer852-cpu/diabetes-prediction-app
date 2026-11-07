# Deploy Karo — Free, GitHub + Streamlit Cloud (10 Minutes)

Ye app maine already test kar li hai, sahi chal rahi hai. Ab isay live website banane ke
sirf ye steps follow karo.

---

## Step 1: GitHub Pe Naya Repository Banao

1. [github.com](https://github.com) pe login karo
2. Upar right corner mein **"+"** icon → **"New repository"**
3. Repository name: `diabetes-risk-prediction` (ya koi bhi naam)
4. **Public** select karo (Streamlit free tier ke liye public zaroori hai)
5. "Create repository" dabao

---

## Step 2: Is `7_Web_App` Folder Ko GitHub Pe Upload Karo

**Option A — Browser Se (Aasan, Coding Ki Zarurat Nahi):**
1. Apne naye GitHub repository page pe jao
2. "uploading an existing file" link pe click karo
3. `7_Web_App` folder ke ANDAR ki chaaron cheezein drag-drop karo:
   - `app.py`
   - `diabetes_pipeline.py` (app isi file se leakage-free hybrid model import karti hai)
   - `requirements.txt`
   - `data` folder (poora, `diabetes_raw.csv` ke saath)
4. Neeche "Commit changes" dabao

**Option B — Terminal Se (agar Git install hai):**
```bash
cd 7_Web_App
git init
git add .
git commit -m "Diabetes prediction app"
git branch -M main
git remote add origin https://github.com/<aapka-username>/diabetes-risk-prediction.git
git push -u origin main
```

---

## Step 3: Streamlit Cloud Pe Deploy Karo

1. Jao: **[share.streamlit.io](https://share.streamlit.io)**
2. **"Sign in with GitHub"** — apne GitHub account se login karo (permission allow karo)
3. **"Create app"** ya **"New app"** button dabao
4. Ye options fill karo:
   - **Repository:** `<aapka-username>/diabetes-risk-prediction`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. **"Deploy!"** dabao

---

## Step 4: 2-3 Minute Wait Karo

Streamlit khud hi `requirements.txt` dekh kar sari libraries install karega aur app build karega.
Progress logs screen pe dikhengi.

---

## Step 5: Live Link Mil Jayega! 🎉

Kuch aisa URL milega:
```
https://diabetes-risk-prediction-<random-id>.streamlit.app
```

Ye link kisi ko bhi bhej sakte ho — professor, dost, koi bhi — browser mein khul jayegi,
kuch install karne ki zarurat nahi unhe.

---

## Agar Deploy Karte Waqt Error Aaye

| Error | Wajah | Solution |
|---|---|---|
| `ModuleNotFoundError` | Koi library `requirements.txt` mein missing | Us library ka naam `requirements.txt` mein add karo |
| `FileNotFoundError: data/diabetes_raw.csv` | Data folder upload nahi hua | Step 2 dobara karo, `data` folder zaroor include karo |
| App bohot slow load ho rahi | Pehli baar model train ho raha hai (normal) | 1-2 min wait karo, agli baar cache se fast chalega |
| "App is sleeping" | Free tier pe app kuch der na use ho toh so jati hai | Bas link kholo, khud jaag jayegi (30-60 sec lagega) |

---

## Baad Mein App Update Karni Ho Toh?

GitHub repository mein file edit karo (ya naye version upload karo) → "Commit changes" dabao
→ Streamlit Cloud khud hi automatically naya version deploy kar dega, kuch aur nahi karna.
