#!/bin/bash

# Push Aegis to YOUR GitLab Repository
# Repository: https://gitlab.com/sukin/aegis

echo "================================================"
echo "🚀 PUSHING AEGIS TO GITLAB"
echo "================================================"
echo ""
echo "Your repository: https://gitlab.com/sukin/aegis"
echo ""

# Option 1: Using HTTPS (you'll need to enter username/password)
echo "📝 To push with HTTPS (enter your GitLab username & password):"
echo ""
echo "git push -u origin main"
echo ""

# Option 2: Using SSH (recommended for easier future pushes)
echo "🔑 Or set up SSH (recommended):"
echo ""
echo "1. First, add your SSH key to GitLab:"
echo "   cat ~/.ssh/id_rsa.pub"
echo "   (Copy this and add to GitLab.com → Settings → SSH Keys)"
echo ""
echo "2. Change remote to SSH:"
echo "   git remote set-url origin git@gitlab.com:sukin/aegis.git"
echo ""
echo "3. Then push:"
echo "   git push -u origin main"
echo ""

# Option 3: Using Personal Access Token
echo "🎫 Or use Personal Access Token (most secure):"
echo ""
echo "1. Go to GitLab.com → Settings → Access Tokens"
echo "2. Create a token with 'write_repository' scope"
echo "3. Push with:"
echo "   git push https://sukin:YOUR_TOKEN@gitlab.com/sukin/aegis.git main"
echo ""

echo "================================================"
echo "📦 What will be pushed (118 files):"
echo "================================================"
echo "✅ Complete Aegis API implementation"
echo "✅ Runtime & training data protection"
echo "✅ Customer dashboard & onboarding"
echo "✅ Patent documentation"
echo "✅ Competitor-safe website"
echo "✅ All deployment scripts"
echo ""
echo "Choose one of the methods above and push your code!"