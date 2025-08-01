from transformers import AutoTokenizer
import os

def test_tokenizer_load():
    model_name_or_path = "bert-base-uncased"
    
    try:
        print("Trying local loading...")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
        print("✅ Local load successful:", tokenizer)
    except Exception as e:
        print("⚠️ Local load failed:", e)
        print("Trying online loading...")
        try:
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            print("✅ Online load successful:", tokenizer)
        except Exception as e:
            print("❌ Online load failed:", e)

if __name__ == "__main__":
    test_tokenizer_load()
