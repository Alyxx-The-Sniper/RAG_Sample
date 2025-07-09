from fastapi import FastAPI, Body
import uvicorn
from retrieval import retrieve, call_deepinfra_llm

app = FastAPI()

fallback_answer = """
Sorry, I can’t answer that—it’s beyond my current knowledge base right now.
I’m an AI bot trained to answer questions on a variety of topics, including:

- Personal Data (Limited info only)
- Artificial Intelligence: https://voice-time-capsule-production.up.railway.app
- Machine Learning: https://medium.com/@kaikuh/machine-learning-code-focus-1bf13c848bc7
- Data Science: https://medium.com/@kaikuh/anomaly-detection-shap-bd352438987f , https://medium.com/@kaikuh/statistical-analysis-part-2-3-b14b87f85abd
- Foods, Travel, Movies, and more.

Please feel free to connect with me on LinkedIn for more info:
https://www.linkedin.com/in/alexis-mandario-b546881a8/

Note: To ensure my answers are based on the ground truth, I’ve set the similarity search to an appropriate value.
""".lstrip()


@app.post("/chat")
async def chat(query: str = Body(..., embed=True)):
    # Step 1: Retrieve top-k relevant context from Pinecone
    paired = await retrieve(query)

    if not paired:
        return {"response": fallback_answer}

    # Combine context into a single string
    context_str = "\n".join(f"Q: {q}\nA: {a}" for q, a in paired) # paired is a list

    # Step 2: Generate answer from LLM
    response = await call_deepinfra_llm(query, context_str)
    # print(f"LLM Response: ok")

    return {"response": response}


if __name__ == "__main__":
    uvicorn.run("this_module_name:app", host="0.0.0.0", port=8000, reload=True)
