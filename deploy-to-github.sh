#!/bin/bash

echo "🚀 Deploying SafeKey Lab to GitHub Pages..."
echo "GitHub User: sukin-safekeylab"
echo ""

# Create/update the repository on GitHub
echo "📦 Preparing deployment..."

# Stage all changes
git add .

# Commit
git commit -m "Deploy SafeKey Lab website" || echo "No changes to commit"

# Push to the new repository
echo "⬆️ Pushing to GitHub (sukin-safekeylab/safekeylab)..."
git push safekeylab main || git push safekeylab master:main

echo ""
echo "✅ Deployment initiated!"
echo ""
echo "📋 IMPORTANT - Manual steps required:"
echo ""
echo "1. Go to: https://github.com/sukin-safekeylab/safekeylab"
echo "   (Create the repository if it doesn't exist yet)"
echo ""
echo "2. Go to Settings → Pages"
echo "   - Source: GitHub Actions"
echo "   - Save"
echo ""
echo "3. Your site will be live at:"
echo "   🌐 https://sukin-safekeylab.github.io/safekeylab/"
echo ""
echo "4. To use safekeylab.com:"
echo "   - Add these DNS A records:"
echo "     • 185.199.108.153"
echo "     • 185.199.109.153"
echo "     • 185.199.110.153"
echo "     • 185.199.111.153"
echo ""
echo "✨ After GitHub Actions runs, your site will be live!"