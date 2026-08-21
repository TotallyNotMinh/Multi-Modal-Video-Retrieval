import os
import sys
import argparse
import time
from pyngrok import ngrok, conf

def main():
    parser = argparse.ArgumentParser(description="Share AIC 2026 Studio via ngrok tunnel")
    parser.add_argument("--port", type=int, default=8080, help="Local port to expose (default: 8080)")
    parser.add_argument("--token", type=str, default="", help="Ngrok Authtoken (optional if NGROK_AUTHTOKEN is set)")
    args = parser.parse_args()

    token = args.token or os.environ.get("NGROK_AUTHTOKEN", "")
    if token:
        ngrok.set_auth_token(token)
        print("🔑 Ngrok authtoken configured.")

    try:
        ngrok.kill()
        tunnel = ngrok.connect(f"127.0.0.1:{args.port}", bind_tls=True)
        print("\n" + "=" * 60)
        print("🚀 AIC 2026 Studio Public Tunnel Active!")
        print(f"👉 Public URL: {tunnel.public_url}")
        print(f"👉 Local URL : http://127.0.0.1:{args.port}")
        print("=" * 60 + "\n")
        print("Press Ctrl+C to stop the tunnel.")

        # Keep alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down ngrok tunnel...")
        ngrok.kill()
    except Exception as e:
        print(f"\n❌ Error starting ngrok tunnel: {e}")
        print("\n💡 Tip: Ngrok requires a free authtoken. Get one at https://dashboard.ngrok.com/get-started/your-authtoken")
        print("Then run:")
        print(f"  python scripts/share_ngrok.py --token YOUR_NGROK_TOKEN\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
