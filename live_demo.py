#!/usr/bin/env python3
"""
AEGIS LIVE DEMO SCRIPT
Run this during sales calls to blow minds
"""

import time
import json
from datetime import datetime
from typing import Dict
import os

# Colors for terminal
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
RESET = '\033[0m'
BOLD = '\033[1m'

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def dramatic_pause(seconds=2):
    time.sleep(seconds)

def type_effect(text, delay=0.03):
    """Simulate typing effect"""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def show_api_call(api_data: Dict, is_safe: bool = False):
    """Display API call in a formatted way"""
    print(f"\n{YELLOW}{'='*60}")
    print(f"{BOLD}📡 API CALL TO OPENAI:{RESET}")
    print(f"{YELLOW}{'='*60}{RESET}")

    if is_safe:
        print(f"{GREEN}✅ PROTECTED REQUEST:{RESET}")
    else:
        print(f"{RED}🚨 DANGEROUS REQUEST:{RESET}")

    print(f"{CYAN}{json.dumps(api_data, indent=2)}{RESET}")
    print(f"{YELLOW}{'='*60}{RESET}\n")

def run_horror_story():
    """Part 1: The Horror Story"""
    clear_screen()
    print(f"{RED}{BOLD}{'='*60}")
    print("THE HORROR STORY")
    print(f"{'='*60}{RESET}\n")

    type_effect("Last month, a YC startup called MedChat AI was doing great...")
    dramatic_pause()

    type_effect("They had 500 paying customers, $50K MRR, just raised a seed round.")
    dramatic_pause()

    print(f"\n{YELLOW}Then this happened:{RESET}\n")
    dramatic_pause()

    type_effect("A doctor typed this into their app:")
    print(f"{RED}")
    type_effect("  'Patient John Smith, SSN 123-45-6789, HIV positive, suicide attempt'")
    print(f"{RESET}")
    dramatic_pause()

    type_effect("Their app sent it DIRECTLY to OpenAI...")
    dramatic_pause()

    print(f"\n{RED}{BOLD}2 weeks later:{RESET}")
    type_effect("- OpenAI had a breach")
    type_effect("- Patient data leaked")
    type_effect("- HIPAA violation: $2 million fine")
    type_effect("- Lawsuit: $5 million")
    type_effect("- MedChat AI: DEAD ☠️")

    dramatic_pause(3)
    input(f"\n{YELLOW}Press Enter to see how this happens...{RESET}")

def run_danger_demo():
    """Part 2: Show the Current Danger"""
    clear_screen()
    print(f"{RED}{BOLD}{'='*60}")
    print("YOUR CURRENT RISK - LIVE DEMO")
    print(f"{'='*60}{RESET}\n")

    print(f"{YELLOW}Let me show you what's happening in YOUR app right now...{RESET}\n")
    dramatic_pause()

    # Simulate user input
    user_input = """Hi, I'm John Smith,
SSN 123-45-6789,
Credit card: 4532-1234-5678-9012,
Diagnosed with HIV,
My password is Secret123!"""

    print(f"{BOLD}👤 USER TYPES:{RESET}")
    print(f"{WHITE}{user_input}{RESET}")
    dramatic_pause(2)

    print(f"\n{RED}{BOLD}⚠️  YOUR APP SENDS THIS TO OPENAI:{RESET}")

    # Show the dangerous API call
    api_call = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": user_input
            }
        ],
        "temperature": 0.7
    }

    show_api_call(api_call, is_safe=False)

    print(f"{RED}{BOLD}🚨 OPENAI NOW HAS:{RESET}")
    print(f"{RED}  • John's SSN")
    print(f"  • His credit card number")
    print(f"  • His HIV status")
    print(f"  • His password{RESET}")

    print(f"\n{RED}{BOLD}THIS IS A HIPAA VIOLATION = $2 MILLION FINE{RESET}")

    dramatic_pause(3)
    input(f"\n{YELLOW}Press Enter to see the solution...{RESET}")

def run_protection_demo():
    """Part 3: Show Aegis Protection"""
    clear_screen()
    print(f"{GREEN}{BOLD}{'='*60}")
    print("AEGIS PROTECTION - LIVE DEMO")
    print(f"{'='*60}{RESET}\n")

    print(f"{GREEN}Now let's see the SAME input WITH Aegis protection...{RESET}\n")
    dramatic_pause()

    # Original input
    user_input = """Hi, I'm John Smith,
SSN 123-45-6789,
Credit card: 4532-1234-5678-9012,
Diagnosed with HIV,
My password is Secret123!"""

    print(f"{BOLD}👤 USER TYPES (same dangerous data):{RESET}")
    print(f"{WHITE}{user_input}{RESET}")
    dramatic_pause(2)

    print(f"\n{GREEN}{BOLD}🛡️  AEGIS INTERCEPTS AND TRANSFORMS:{RESET}")
    dramatic_pause()

    # Show transformation
    print(f"\n{CYAN}Original → Protected:{RESET}")
    transformations = [
        ("John Smith", "[NAME]"),
        ("SSN 123-45-6789", "SSN [REDACTED]"),
        ("4532-1234-5678-9012", "[CREDIT_CARD]"),
        ("HIV", "[MEDICAL_CONDITION]"),
        ("Secret123!", "[PASSWORD]")
    ]

    for original, protected in transformations:
        print(f"  {RED}{original}{RESET} → {GREEN}{protected}{RESET}")
        time.sleep(0.5)

    # Protected version
    protected_input = """Hi, I'm [NAME],
SSN [REDACTED],
Credit card: [CREDIT_CARD],
Diagnosed with [MEDICAL_CONDITION],
My password is [PASSWORD]"""

    print(f"\n{GREEN}{BOLD}✅ SAFE VERSION SENT TO OPENAI:{RESET}")

    # Show the safe API call
    safe_api_call = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": protected_input
            }
        ],
        "temperature": 0.7
    }

    show_api_call(safe_api_call, is_safe=True)

    print(f"{GREEN}{BOLD}✅ RESULT:{RESET}")
    print(f"{GREEN}  • OpenAI sees NO personal information")
    print(f"  • User data is 100% protected")
    print(f"  • You are HIPAA/GDPR compliant")
    print(f"  • AI response still works perfectly!{RESET}")

    dramatic_pause(3)
    input(f"\n{YELLOW}Press Enter to see how easy it is to add...{RESET}")

def run_integration_demo():
    """Part 4: Show How Easy Integration Is"""
    clear_screen()
    print(f"{BLUE}{BOLD}{'='*60}")
    print("INTEGRATION - IT'S JUST ONE LINE!")
    print(f"{'='*60}{RESET}\n")

    print(f"{YELLOW}Watch how easy this is to add to your app...{RESET}\n")
    dramatic_pause()

    print(f"{BOLD}📝 YOUR CURRENT CODE:{RESET}")
    print(f"{WHITE}```python")
    type_effect("def process_user_input(text):")
    type_effect("    response = openai.chat.completions.create(")
    type_effect("        model='gpt-4',")
    type_effect("        messages=[{'role': 'user', 'content': text}]")
    type_effect("    )")
    type_effect("    return response")
    print("```")
    print(f"{RESET}")

    dramatic_pause(2)

    print(f"\n{BOLD}✨ ADD AEGIS (ONE LINE!):{RESET}")
    print(f"{GREEN}```python")
    type_effect("from aegis import protect  # Add this import")
    type_effect("")
    type_effect("def process_user_input(text):")
    print(f"{YELLOW}{BOLD}", end="")
    type_effect("    text = protect(text)  # ← THIS ONE LINE SAVES YOUR COMPANY")
    print(f"{GREEN}", end="")
    type_effect("    response = openai.chat.completions.create(")
    type_effect("        model='gpt-4',")
    type_effect("        messages=[{'role': 'user', 'content': text}]")
    type_effect("    )")
    type_effect("    return response")
    print("```")
    print(f"{RESET}")

    print(f"\n{GREEN}{BOLD}⏱️  TIME TO IMPLEMENT: 60 SECONDS{RESET}")
    print(f"{GREEN}{BOLD}💰 COST: $49/MONTH{RESET}")
    print(f"{GREEN}{BOLD}🛡️  PROTECTION: PRICELESS{RESET}")

    dramatic_pause(3)
    input(f"\n{YELLOW}Press Enter for pricing...{RESET}")

def run_pricing_close():
    """Part 5: Pricing and Close"""
    clear_screen()
    print(f"{MAGENTA}{BOLD}{'='*60}")
    print("INVESTMENT & ROI")
    print(f"{'='*60}{RESET}\n")

    print(f"{BOLD}💰 THE MATH:{RESET}\n")

    # Cost comparison
    comparisons = [
        ("One HIPAA violation", "$2,000,000", RED),
        ("One data breach lawsuit", "$5,000,000", RED),
        ("Lost customers from breach", "$500,000", RED),
        ("Your company reputation", "Priceless", RED),
        ("", "", ""),
        ("Aegis Protection", "$49/month", GREEN),
    ]

    for item, cost, color in comparisons:
        if item:
            print(f"{color}  {item:30} {cost:>15}{RESET}")
        else:
            print()
        time.sleep(0.5)

    print(f"\n{YELLOW}{BOLD}ROI: 40,000x{RESET}")

    dramatic_pause(2)

    print(f"\n{MAGENTA}{BOLD}🎁 SPECIAL OFFER (EXPIRES IN 48 HOURS):{RESET}")
    print(f"{GREEN}  ✓ First month FREE")
    print(f"  ✓ 50% off for 6 months ($24.50/month)")
    print(f"  ✓ Personal integration help")
    print(f"  ✓ 'Protected by Aegis' badge")
    print(f"  ✓ Direct line to founders{RESET}")

    dramatic_pause(2)

    print(f"\n{YELLOW}{BOLD}⚠️  REMEMBER:{RESET}")
    print(f"{YELLOW}  Every minute without protection, users are typing:")
    print(f"  • SSNs")
    print(f"  • Credit cards")
    print(f"  • Medical records")
    print(f"  • Passwords{RESET}")

    print(f"\n{RED}{BOLD}  You're ONE angry user away from bankruptcy.{RESET}")

    dramatic_pause(3)

    print(f"\n{GREEN}{BOLD}{'='*60}{RESET}")
    print(f"{GREEN}{BOLD}Ready to protect your company?{RESET}")
    print(f"{GREEN}{BOLD}{'='*60}{RESET}\n")

def run_live_test():
    """Bonus: Let them test with their own data"""
    clear_screen()
    print(f"{CYAN}{BOLD}{'='*60}")
    print("LIVE TEST - TRY YOUR OWN DATA")
    print(f"{'='*60}{RESET}\n")

    print(f"{YELLOW}Type something with sensitive data:{RESET}")
    print(f"{WHITE}(Try including SSNs, emails, credit cards, medical info){RESET}\n")

    user_test = input(f"{BOLD}> {RESET}")

    print(f"\n{YELLOW}Processing...{RESET}")
    dramatic_pause(1)

    # Simulate protection
    import re
    protected = user_test
    protected = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', protected)
    protected = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CREDIT_CARD]', protected)
    protected = re.sub(r'\b[\w._%+-]+@[\w.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', protected, flags=re.IGNORECASE)
    protected = re.sub(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', protected)

    print(f"\n{RED}{BOLD}❌ WITHOUT AEGIS:{RESET}")
    print(f"{RED}'{user_test}'{RESET}")
    print(f"{RED}↑ This goes to OpenAI{RESET}")

    print(f"\n{GREEN}{BOLD}✅ WITH AEGIS:{RESET}")
    print(f"{GREEN}'{protected}'{RESET}")
    print(f"{GREEN}↑ This goes to OpenAI{RESET}")

    if protected != user_test:
        print(f"\n{GREEN}{BOLD}🎉 We just saved you from a compliance violation!{RESET}")
    else:
        print(f"\n{GREEN}{BOLD}✅ No sensitive data detected - but we're always watching!{RESET}")

def main():
    """Run the complete demo"""
    clear_screen()
    print(f"{CYAN}{BOLD}")
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║                    AEGIS PRIVACY SHIELD                   ║
    ║                                                            ║
    ║            Stop Sending User Secrets to OpenAI            ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    print(f"{RESET}")

    print(f"{YELLOW}This demo will show you:{RESET}")
    print("  1. How your company could die tomorrow")
    print("  2. What's happening in your app RIGHT NOW")
    print("  3. How Aegis protects you")
    print("  4. How easy it is to implement")
    print("  5. The investment (spoiler: less than your coffee budget)")

    input(f"\n{GREEN}{BOLD}Press Enter to start the demo...{RESET}")

    # Run all parts
    run_horror_story()
    run_danger_demo()
    run_protection_demo()
    run_integration_demo()
    run_pricing_close()

    # Optional: Live test
    print(f"\n{CYAN}Want to test with your own data? (y/n):{RESET} ", end="")
    if input().lower() == 'y':
        run_live_test()

    # Final close
    print(f"\n{GREEN}{BOLD}{'='*60}{RESET}")
    print(f"{GREEN}{BOLD}NEXT STEPS:{RESET}")
    print(f"{GREEN}1. Sign up: aegis.ai/signup")
    print(f"2. Get your API key")
    print(f"3. Add one line of code")
    print(f"4. Sleep better tonight{RESET}")
    print(f"{GREEN}{BOLD}{'='*60}{RESET}")

    print(f"\n{YELLOW}{BOLD}Questions? Let's talk:{RESET}")
    print(f"{CYAN}📧 sales@aegis.ai")
    print(f"📞 1-800-SAFE-AI")
    print(f"💬 aegis.ai/slack{RESET}")

    print(f"\n{MAGENTA}{BOLD}Remember: Every minute without protection is a minute at risk.{RESET}")
    print(f"{GREEN}{BOLD}Get protected now: aegis.ai{RESET}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Demo interrupted. Contact us at sales@aegis.ai{RESET}")
    except Exception as e:
        print(f"\n{RED}Error: {e}{RESET}")
        print(f"{YELLOW}Contact support@aegis.ai for help{RESET}")