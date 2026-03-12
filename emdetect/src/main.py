import sys
import os

base_path   = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(base_path, 'config.json')
sys.path.append(base_path)

try:
    from ml.engine import PsychologyEngine
    from ml.loader import MarkLoader
except ImportError as e:
    print(f"Critical import error: {e}")
    print(f"Make sure the 'ml' folder is at: {base_path}")
    sys.exit(1)


def detect_lang(text: str) -> str:
    # Detect language via langdetect
    try:
        from langdetect import detect
        lang = detect(text)
        return lang if lang in {'ru', 'uk', 'en', 'de'} else 'ru'
    except Exception:
        return 'ru'


def main():
    print("\n" + "=" * 65)
    print("   EMDETECT v2.1: MULTILINGUAL SEMANTIC ANALYSER")
    print("=" * 65)

    try:
        print("Loading MiniLM L12...")
        engine = PsychologyEngine(data_dir=os.path.join(base_path, "data"))
        loader = MarkLoader(engine)

        print(f"Loading config: {config_path}")
        tags_data = loader.load_tags_from_config(config_path)

        if not tags_data:
            print("[!] Error: config is empty or marker files not found.")
            return

        print(f"Ready. Categories loaded: {len(tags_data)}")
        print("Supported languages: Russian, Ukrainian, English, German")
        print("(Type 'exit' / 'выход' / 'beenden' to quit)\n")

    except Exception as e:
        print(f"Startup error: {e}")
        return

    while True:
        try:
            user_text = input("Tell me how do you feel: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down. Take care!")
            break

        if user_text.lower() in {'выход', 'exit', 'quit', 'вихід', 'beenden'}:
            print("Shutting down. Take care!")
            break

        if not user_text:
            continue

        lang = detect_lang(user_text)
        print(f"[{lang.upper()}] Processing...")

        try:
            results = engine.get_top_matches(user_text, lang, tags_data)

            if results:
                print(f"\n{'CATEGORY':<28} {'INTENSITY':<22} SCORE")
                print("-" * 65)
                for res in results:
                    score = res['score']
                    bar   = "█" * int(score * 20)
                    if score > 0.85:
                        label = "(High)  "
                    elif score > 0.70:
                        label = "(Medium)"
                    else:
                        label = "(Low)   "
                    print(f"{res['tag_id']:<28} {bar:<22} {score:.4f} {label}")
                print()
            else:
                print("  > No markers detected. Try describing your state in more detail.\n")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
