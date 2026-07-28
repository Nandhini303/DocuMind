# DocuMind AI — Setup & Deployment Guide (Free Stack)

An AI-powered document analyzer. Paste text or upload a `.txt`/`.pdf`, choose
**Summarize**, **Key Points**, or **Simplify**, and watch the AI's response
stream in live.

- **Frontend:** HTML/CSS/JS (no framework, no build step) — served as static files
- **Backend:** Python FastAPI, streams responses via Server-Sent Events (SSE)
- **LLM:** Anthropic Claude API, called only from the backend
- **Container:** Single Dockerfile
- **Hosting:** Render.com (free web service tier, no credit card required)

---

## Cost summary (this whole project is designed to run at $0)

| Piece | Cost |
|---|---|
| Anthropic API | Free — new Console accounts get a one-time **$5 trial credit** (no card, just phone verification), which comfortably covers building, testing, and demoing this project |
| Hosting | Free — Render's free Web Service tier requires no credit card |
| Domain/HTTPS | Free — Render issues a public `https://*.onrender.com` URL automatically |

**The only real limitation:** Render's free tier spins the service down after
15 minutes of no traffic, and the next request takes 30–60 seconds to "wake
up." That's fine for a class project — just open the URL a minute before
anyone needs to look at it.

> **Note on the assignment brief:** the original brief asks for deployment to
> AWS. Render was substituted specifically to keep this project genuinely
> free of charge (AWS App Runner has no real free tier and Elastic Beanstalk
> can still incur small charges depending on account type). Document this
> substitution and the reasoning in your submission — see the Project Report
> for suggested wording.

---

## 1. Run it locally first (no cost, no deployment needed to test)

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your real ANTHROPIC_API_KEY
```

Get your key at **console.anthropic.com** → sign up (no card needed) →
**API Keys** → **Create Key**. New accounts get $5 in free trial credit
automatically.

```bash
uvicorn main:app --reload --port 8080
```

Open **http://localhost:8080**, paste some text, click **Analyze**. Text
should stream in within a couple of seconds. Check
`http://localhost:8080/api/health` any time to confirm the key is loaded.

---

## 2. Test with Docker locally (optional but recommended)

```bash
cd backend
docker build -t documind-ai .
docker run -p 8080:8080 --env-file .env documind-ai
```

Visit **http://localhost:8080** again — this is exactly what Render will run.

---

## 3. Push your code to GitHub

Render deploys from a Git repository.

```bash
cd documind
git init
git add .
git commit -m "DocuMind AI"
```

Create a new empty repo on github.com, then:

```bash
git remote add origin https://github.com/<your-username>/documind-ai.git
git branch -M main
git push -u origin main
```

`.env` will **not** be pushed — it's already listed in `.gitignore`.

---

## 4. Deploy on Render (free, no card)

1. Sign up at **render.com** with GitHub.
2. Dashboard → **New +** → **Web Service**.
3. Connect your `documind-ai` repo.
4. Render should auto-detect the `Dockerfile`. If it doesn't:
   - **Root Directory:** `backend`
   - **Runtime:** Docker
5. **Instance Type:** Free
6. **Environment Variables** → **Add Environment Variable**:
   - Key: `ANTHROPIC_API_KEY`
   - Value: your real key
7. Click **Create Web Service**. Render builds the image and deploys it —
   first build takes a few minutes.
8. You'll get a public URL like `https://documind-ai.onrender.com`.

### Alternative: deploy via the render.yaml blueprint
This repo includes `backend/render.yaml`. In the Render dashboard you can
use **New + → Blueprint**, point it at your repo, and Render will read the
service definition automatically — you'll just need to fill in the
`ANTHROPIC_API_KEY` value when prompted.

---

## 5. Verify the live deployment

- Visit `<your-url>/api/health` → confirm `"api_key_configured": true`
- Run a real analysis (all 3 modes) on the live URL
- Test on a phone browser too (mobile responsiveness requirement)
- **Wake the app before a demo/grading** — visit the URL ~1 minute early so
  it's out of its sleep state

---

## 6. Security checklist before submission

- [ ] `.env` is in `.gitignore` and was never committed
- [ ] No API key appears anywhere in `static/index.html` or any frontend file
- [ ] The key is set only via Render's Environment Variables panel
- [ ] `/api/health` on the live URL does **not** expose the key value, only
      whether it's configured

---

## 7. Project structure

```
documind/
├── README.md                 ← this file
└── backend/
    ├── main.py                ← FastAPI app + Claude streaming endpoint
    ├── requirements.txt
    ├── Dockerfile
    ├── render.yaml             ← optional Render Blueprint definition
    ├── .env.example
    ├── .gitignore
    └── static/
        └── index.html          ← frontend (served by FastAPI)
```
