# 🚀 GitHub Pages Deployment Guide for SafeKey Lab

## Why GitHub Pages Instead of Vercel?

### ✅ Advantages:
- **100% FREE** - No limits, no pricing tiers
- **No build step needed** - Your site is already static HTML
- **Custom domain included** - safekeylab.com works perfectly
- **Automatic HTTPS** - SSL certificate provided free
- **No account signup** - Just push to GitHub
- **Integrated CI/CD** - GitHub Actions handles everything
- **Better for demos** - No serverless functions needed

---

## 📋 Step-by-Step Deployment

### Step 1: Push Your Code to GitHub
```bash
# If not already a git repo
git init
git add .
git commit -m "Initial SafeKey Lab platform"

# Create repo on GitHub (or use GitHub CLI)
gh repo create safekeylab/safekeylab --public --source=. --remote=origin --push

# Or if repo exists
git remote add origin https://github.com/YOUR_USERNAME/safekeylab.git
git push -u origin main
```

### Step 2: Enable GitHub Pages
1. Go to your repo on GitHub
2. Click **Settings** → **Pages** (left sidebar)
3. Under **Source**, select: **GitHub Actions**
4. Save

### Step 3: Deploy Automatically
The GitHub Action we created will:
- Trigger on every push to `main`
- Deploy the `/website` folder
- Make it available at your domain

Push any change to trigger:
```bash
git add .
git commit -m "Deploy to GitHub Pages"
git push
```

### Step 4: Configure Custom Domain
1. In GitHub repo → **Settings** → **Pages**
2. Under **Custom domain**, enter: `safekeylab.com`
3. Check **Enforce HTTPS**

### Step 5: Update DNS (at your domain registrar)
Add these records to your domain:

#### For apex domain (safekeylab.com):
```
Type: A
Name: @
Value: 185.199.108.153
```
```
Type: A
Name: @
Value: 185.199.109.153
```
```
Type: A
Name: @
Value: 185.199.110.153
```
```
Type: A
Name: @
Value: 185.199.111.153
```

#### For www subdomain (www.safekeylab.com):
```
Type: CNAME
Name: www
Value: YOUR_USERNAME.github.io
```

---

## 🌐 Your URLs

After deployment, your site will be available at:
- **GitHub URL**: `https://YOUR_USERNAME.github.io/safekeylab/`
- **Custom Domain**: `https://safekeylab.com`
- **Dashboard**: `https://safekeylab.com/dashboard.html`
- **API Docs**: `https://safekeylab.com/api-reference.html`

---

## 📁 Repository Structure for GitHub Pages

```
safekeylab/
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions deployment
├── website/                    # This folder gets deployed
│   ├── CNAME                  # Custom domain config
│   ├── index.html             # Homepage
│   ├── dashboard.html         # Main dashboard
│   ├── pricing.html           # Pricing page
│   ├── docs.html              # Documentation
│   ├── css/                   # Styles
│   ├── js/                    # JavaScript
│   └── assets/                # Images/icons
├── README.md                   # Repo documentation
└── (other backend files)       # Not deployed
```

---

## 🔧 Manual Deployment (Alternative)

If you prefer manual deployment:

1. Go to **Settings** → **Pages**
2. Source: **Deploy from a branch**
3. Branch: **main**
4. Folder: **/website**
5. Save

---

## 🎯 Benefits Over Vercel

| Feature | GitHub Pages | Vercel |
|---------|--------------|--------|
| **Price** | Free forever | Free tier limited |
| **Custom Domain** | Free | Free |
| **SSL** | Free | Free |
| **Build Required** | No | Yes |
| **Setup Time** | 2 minutes | 10 minutes |
| **API Routes** | No* | Yes |
| **Analytics** | Basic | Advanced |

*Your API runs separately on your server/cloud, not needed for demo

---

## 🚨 Important Notes

1. **API Backend**: Your Python API (`aegis_real_api.py`) needs separate hosting:
   - Use Render.com (free tier)
   - Use Railway.app ($5/month)
   - Use DigitalOcean App Platform ($5/month)
   - Or keep it local for demos

2. **Static Assets Only**: GitHub Pages only serves static files (HTML, CSS, JS)

3. **Update Workflow**: Every push to `main` auto-deploys

---

## 📊 Monitoring Your Deployment

Check deployment status:
1. Go to repo → **Actions** tab
2. See "Deploy to GitHub Pages" workflow
3. Green check = Successfully deployed

---

## 🆘 Troubleshooting

### Site not showing?
- Wait 10 minutes for DNS propagation
- Check Actions tab for errors
- Ensure CNAME file exists in /website

### Custom domain not working?
- Verify DNS records with: `dig safekeylab.com`
- May take up to 24 hours for DNS propagation

### 404 errors?
- Check file paths (case-sensitive)
- Ensure files are in /website folder

---

## 🎉 That's It!

Your SafeKey Lab platform is now live on GitHub Pages:
- **Zero cost**
- **Automatic deployment**
- **Custom domain with HTTPS**
- **No build process needed**

Just push to GitHub and your changes go live automatically!