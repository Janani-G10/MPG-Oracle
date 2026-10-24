# Deploying MPG Oracle on Streamlit

## Project layout
```
mpg-oracle/
├── app.py
├── auto-mpg.csv
├── requirements.txt
└── DEPLOY.md
```

## 1. Run locally first
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py            # opens http://localhost:8501
```

## 2. Push to GitHub
```bash
git init
git add app.py auto-mpg.csv requirements.txt DEPLOY.md
git commit -m "MPG Oracle"
git branch -M main
git remote add origin https://github.com/<your-username>/mpg-oracle.git
git push -u origin main
```

## 3. Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** -> **Deploy a public app from GitHub**.
3. Repository: `<your-username>/mpg-oracle`, Branch: `main`, Main file path: `app.py`.
4. Click **Deploy**. The first build installs requirements (1-3 minutes).
5. Your app gets a public URL like `https://<name>.streamlit.app`.

## Troubleshooting
- **FileNotFoundError / upload prompt**: `auto-mpg.csv` must be in the repo root, next to `app.py`.
- **ModuleNotFoundError**: add the missing package to `requirements.txt`, commit, push; the app redeploys automatically.
- **Slow first load**: models train once, then are cached.
