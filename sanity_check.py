import os
import google.generativeai as genai

print("GENERATION_MODEL =", os.getenv("GENERATION_MODEL"))
print("EMBEDDING_MODEL  =", os.getenv("EMBEDDING_MODEL"))

genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))

# Embedding
emb = genai.embed_content(model=os.getenv("EMBEDDING_MODEL", "models/text-embedding-004"),
                          content="hello embeddings")
print("Embed len:", len(emb["embedding"]))

# Generation
model = genai.GenerativeModel(os.getenv("GENERATION_MODEL", "models/gemini-1.5-flash"))
resp = model.generate_content("Say hi in one short sentence.")
print("Gen:", resp.text)
