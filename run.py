#!/usr/bin/env python3
import sys
import uvicorn
from backend.config import HOST, PORT

def main():
    print("=" * 60)
    print("           A L E T H E A I   W I - F I   R A D A R           ")
    print("      Standard-Compliant BFI Passive Sensing Tracker        ")
    print("=" * 60)
    print(f"[*] Starting FastAPI backend web server...")
    print(f"[*] Access the Web Dashboard at: http://localhost:{PORT}")
    print(f"[*] Press Ctrl+C to terminate.")
    print("-" * 60)
    
    try:
        uvicorn.run("backend.app:app", host=HOST, port=PORT, reload=False, log_level="info")
    except KeyboardInterrupt:
        print("\n[*] Shutting down Aletheai. Goodbye!")
    except Exception as e:
        print(f"\n[!] Server error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
