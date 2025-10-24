#!/bin/bash

# SafeKey Lab - GitHub Pages Deployment Script
# This script will deploy your site to GitHub Pages

echo "🚀 Deploying SafeKey Lab to GitHub Pages..."

# Initialize git if needed
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    git branch -M main
fi

# Add all files
echo "📦 Adding files to git..."
git add .

# Commit changes
echo "💾 Committing changes..."
git commit -m "Deploy SafeKey Lab platform to GitHub Pages" || echo "No changes to commit"

# Add remote if it doesn't exist
if ! git remote | grep -q "origin"; then
    echo "🔗 Adding GitHub remote..."
    git remote add origin https://github.com/sukin-safekeylab/safekeylab.git
fi

# Push to GitHub
echo "⬆️ Pushing to GitHub..."
git push -u origin main

echo "✅ Code pushed to GitHub!"
echo ""
echo "📋 Next steps:"
echo "1. Go to: https://github.com/sukin-safekeylab/safekeylab/settings/pages"
echo "2. Under 'Source', select 'GitHub Actions'"
echo "3. Save the settings"
echo ""
echo "🌐 Your site will be available at:"
echo "   https://sukin-safekeylab.github.io/safekeylab/"
echo "   or"
echo "   https://safekeylab.com (after DNS setup)"
echo ""
echo "✨ Deployment complete!"