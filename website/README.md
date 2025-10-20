# Aegis Website Versions

## 🔒 Production Version (`index-safe.html`)
**USE THIS FOR PUBLIC DEPLOYMENT**

This is the competitor-safe version that should be deployed publicly. It:
- ✅ Hides all technical implementation details
- ✅ Removes specific performance metrics
- ✅ Keeps benefits vague but compelling
- ✅ No code examples or API documentation
- ✅ Focus on problem/solution, not HOW it works
- ✅ "Request Demo" instead of self-service

### What's Protected:
- Auto-configuration algorithms
- Multi-head attention architecture
- Binary search optimization
- Exact performance numbers (35M req/sec)
- 6 privacy methods details
- Integration code examples
- API endpoints and parameters
- Pricing information

## 🔓 Internal Version (`index.html`)
**INTERNAL USE ONLY - DO NOT DEPLOY**

This version contains:
- Full technical details
- Patent innovations
- Performance benchmarks
- Code examples
- API documentation
- Architecture diagrams

Use this version for:
- Internal documentation
- Post-NDA customer discussions
- Technical deep dives after contract signing
- Investor due diligence (under NDA)

## 📁 File Structure

### Public Files (Safe to Deploy):
- `index-safe.html` - Main marketing site
- `docs-gated.html` - Login-protected documentation
- `styles.css` - Styling (safe)
- `aegis-icon.svg` - Logo (safe)

### Private Files (DO NOT DEPLOY):
- `index.html` - Full technical version
- `documentation.html` - Detailed docs
- `api-reference.html` - API details
- `features.html` - Technical features
- `pricing.html` - Pricing strategy
- All other `.html` files

## 🚀 Deployment Instructions

1. **For Production:**
   ```bash
   # Rename safe version to index
   cp index-safe.html index.html

   # Deploy only these files:
   - index.html (the safe version)
   - docs-gated.html
   - styles.css
   - aegis-icon.svg
   ```

2. **Customer Portal (Separate Domain):**
   - Set up `portal.aegis-shield.ai` with authentication
   - Deploy full documentation there
   - Require signed contracts for access

## ⚠️ Security Notes

1. **Never deploy the detailed versions publicly**
2. **Use the safe version for all marketing**
3. **Share technical details only after NDA/contract**
4. **Keep performance numbers vague ("industry-leading")**
5. **Don't mention specific algorithms or methods**

## 💡 Marketing Messaging (Safe to Use)

✅ "Advanced privacy protection"
✅ "Enterprise-grade compliance"
✅ "Seamless integration"
✅ "Industry-leading performance"
✅ "Patented technology"

❌ "35 million requests per second"
❌ "Binary search optimization"
❌ "Multi-head attention detection"
❌ "Six privacy methods"
❌ "Auto-configuration in 10 seconds"

Remember: **Competitors are watching. Protect your IP!**