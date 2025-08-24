from huggingface_hub import InferenceClient
import base64
from langchain_community.llms import HuggingFaceEndpoint
import os
from dotenv import load_dotenv

# Load environment variables from token.env file
load_dotenv("token.env")

# Get HF token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables. Please set it in your token.env file.")
import os
HF_TOKEN = os.getenv("HF_TOKEN")


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-VL-32B-Instruct",       # e.g.
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN
)


# Get image path from user
image_path = input("Enter the path to the image: ").strip().strip('"')

# Convert image to base64
with open(image_path, "rb") as f:
    base64_image = base64.b64encode(f.read()).decode("utf-8")

# Prepare messages correctly
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": base64_image   # ✅ send as "image" key, not "image_url"
            },
            {
                "type": "text",
                "text": "اقرأ النص المكتوب باليد في هذه الصورة بدقة. اكتب النص كما هو تماماً دون أي تغيير."
            }
        ]
    }
]

client = InferenceClient(token=HF_TOKEN)

# Send request
output = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-32B-Instruct",
    messages=messages
)

# Extract text
arabic_text = ""
for item in output.choices[0].message["content"]:
    if item.get("type") == "text":
        arabic_text += item.get("text", "")

# Use LangChain's llm to generate a response
response = llm.invoke(messages)

# Print the result
print("\n Extracted Arabic Text:\n")
print(response)
