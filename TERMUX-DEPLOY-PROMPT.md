# Prompt to paste into an agent on the phone (Termux)

Copy everything inside the block below into your Termux AI agent / or follow it
yourself. It fixes the exact broken state in your screenshot: an empty repo
with only `.git`, a literal `REALREPONAME` remote, and a push that failed with
`src refspec main does not match any` because there were zero commits.

---

```
You are setting up the ClipForge backend on this Android phone in Termux.

Context: the deploy zip is somewhere under /sdcard/Download/, probably named
clipforge-deploy.zip. A previous attempt unzipped it into a nested path ending
in "clipforge-deploy.zip.ZIP" and left ~/clipforge-clean containing only .git,
so `git add .` had nothing to commit and the push failed with
"src refspec main does not match any".

Goal: end with a git repo at ~/clipforge-deploy that contains the real package,
has one initial commit, and is pushed to github.com/dieljean959-afk/clipforge.

Rules:
- Never print, echo, or write a GitHub token anywhere. Ask me for it and read it
  with `read -r -s` only. Never put it in a command line, a file, a log, or the
  git remote URL.
- Do not create or commit any .env, .env.local, .env.keys, cookies.txt, data/,
  outputs/, downloads/, *.mp4, *.log or token file.
- Show me the real output of every command. Do not claim success without it.

Steps:
1. Find the zip: `ls -la /sdcard/Download/ | grep -i clipforge`
2. Extract into a clean absolute path, not a nested one:
   `rm -rf ~/clipforge-deploy && mkdir -p ~/clipforge-deploy`
   `cd ~/clipforge-deploy && unzip -o /sdcard/Download/clipforge-deploy.zip`
   If the file is named *.zip.ZIP, rename it first: `mv "clipforge-deploy.zip.ZIP" clipforge-deploy.zip`
3. Prove the extraction worked before touching git:
   `ls -a` must show Dockerfile, pyproject.toml, app/, worker/, scripts/, .env.example
   `unzip -l /sdcard/Download/clipforge-deploy.zip | tail -3` must report ~79 files
   If `ls -a` shows only .git, the zip went somewhere else — stop and tell me.
4. Delete the broken empty repo: `rm -rf ~/clipforge-clean`
5. Initialize and commit:
   `cd ~/clipforge-deploy`
   `git init -b main`
   `git add -A`
   `git status --short | head -40`   (confirm files are staged, .env is absent)
   `git -c user.name="dieljean" -c user.email="dieljean959@gmail.com" commit -m "ClipForge backend: FastAPI + ffmpeg deployable package"`
   `git log --oneline` must show one commit. If it says "nothing to commit", step 3 failed.
6. Set the real remote (the literal REALREPONAME was a placeholder):
   `git remote add origin https://github.com/dieljean959-afk/clipforge.git`
   (or `git remote set-url origin ...` if origin already exists)
7. Push. Ask me for the token with hidden input, then:
   `read -r -s -p "GitHub token: " GH_TOKEN; echo`
   `git -c credential.helper= push -u origin main` using an env-based askpass or
   a one-shot `https://x-access-token:$GH_TOKEN@github.com/...` URL that you
   immediately revert with `git remote set-url origin https://github.com/dieljean959-afk/clipforge.git`
   and `unset GH_TOKEN`. Confirm with `git remote -v` that no token remains.
8. Report: `git log --oneline`, `git remote -v`, `git status`, and the file count.
```

---

## After the push

1. Open Render → New → Blueprint → choose `dieljean959-afk/clipforge`. The
   `Dockerfile` at the repo root builds the API; Render sets `$PORT`.
2. Add secrets in Render's Environment tab (names only, values never in the repo):
   `DEEPSEEK_API_KEY` (or `GROQ_API_KEY`), plus optional `WORKER_TOKEN`.
3. When Render shows the service live, open `https://<service>.onrender.com/api/health`
   — it must return JSON with `"ok": true`.
4. Give me that URL. It becomes the site's `BACKEND_URL` secret, and `/api/*` on
   the published site starts working from the same origin.
