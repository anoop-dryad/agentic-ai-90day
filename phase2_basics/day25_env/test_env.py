import os

api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    print(f"✅ Key loaded. Length: {len(api_key)} chars")
    print(f"   First 10 chars: {api_key[:10]}...")
else:
    print("❌ Key not found. Run 'direnv allow' in this folder.")