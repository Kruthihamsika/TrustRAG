import streamlit as st
from src.rag import retrieve, build_prompt, generate_with_ollama, compute_confidence

st.set_page_config(page_title="Elite Hybrid Offline RAG", layout="wide")

st.title("🚀 Elite Hybrid Offline RAG Assistant")
st.markdown("Multi-Document Hybrid RAG with Reranking + Confidence Scoring")

query = st.text_input("Ask a question:")

if st.button("Generate Answer") and query:

    with st.spinner("Retrieving context..."):
        contexts = retrieve(query)

    # -------------------------
    # If absolutely no contexts
    # -------------------------
    if not contexts:
        st.error("No context retrieved from documents.")
        st.stop()

    # -------------------------
    # Build prompt
    # -------------------------
    prompt = build_prompt(query, contexts)

    with st.spinner("Generating answer with Ollama..."):
        answer = generate_with_ollama(prompt)

    # -------------------------
    # Display Answer
    # -------------------------
    st.subheader("🧠 Answer")
    st.write(answer)

    # -------------------------
    # Confidence Score
    # -------------------------
    confidence_level, confidence_score = compute_confidence(contexts, answer)

    st.subheader("🔎 Confidence")
    if confidence_level == "HIGH":
        st.success(f"{confidence_level} ({confidence_score})")
    elif confidence_level == "MEDIUM":
        st.warning(f"{confidence_level} ({confidence_score})")
    else:
        st.error(f"{confidence_level} ({confidence_score})")

    # -------------------------
    # Sources
    # -------------------------
    st.subheader("📚 Sources")

    for c in contexts:
        st.markdown(
            f"""
            **{c['source']}**  
            Page: {c['page']}  
            Rerank Score: {c['rerank_score']:.4f}
            ---
            """
        )