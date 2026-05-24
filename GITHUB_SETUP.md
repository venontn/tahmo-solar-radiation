# Publish to GitHub (one-time)

**Author:** Venon Takunda Nyadombo — all commits use your name only (no Cursor co-author).

## Step 1 — Log in to GitHub CLI

Open PowerShell and run:

```powershell
gh auth login
```

Choose:

- **GitHub.com**
- **HTTPS**
- **Login with a web browser** (easiest)

## Step 2 — Create the repo and push

```powershell
cd C:\Users\PC\Projects\tahmo-solar-radiation
.\scripts\publish_github.ps1
```

Your repo will be live at: **https://github.com/venontn/tahmo-solar-radiation**

---

## Alternative (no `gh` CLI)

1. Open https://github.com/new  
2. Repository name: `tahmo-solar-radiation`  
3. **Public** → **Create repository** (do not add README — we already have one)  
4. In PowerShell:

```powershell
cd C:\Users\PC\Projects\tahmo-solar-radiation
git push -u origin main
```

---

## What is uploaded

- Source code (`src/`), Colab notebook, scripts, README, AUTHORS  
- **Not** large CSVs (`Train.csv`, `Test.csv`) — see `data/README.md` for download instructions  
