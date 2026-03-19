
import streamlit as st
import requests

# 1. UI Configuration
st.set_page_config(page_title="Legal Summarizer", page_icon="⚖️")
st.title("⚖️ Legal Multi-Doc Summarizer")

# 2. Sidebar Settings
with st.sidebar:
    st.header("Settings")
    summary_length = st.radio("Target Length:", ("short", "long"), index=0)
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    submit_btn = st.button("Generate Summary")

# 3. Main Logic
# 3. Main Logic
if submit_btn:
    if not uploaded_files:
        st.error("Please upload files.")
    else:
        # Prepare the files for the request
        files_payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
        
        # JUST the spinner - everything happens inside this block
        with st.spinner(f"AI is analyzing documents and generating your {summary_length} summary..."):
            try:
                res = requests.post(
                    "http://127.0.0.1:8000/summarize", 
                    files=files_payload, 
                    data={"length": summary_length},
                    timeout=1200 
                )
                
                if res.status_code == 200:
                    st.success("Analysis Complete!")
                    st.subheader("Final Summary")
                    st.markdown(res.json()["summary"])
                else:
                    st.error(f"Backend Error: {res.json().get('error', 'Unknown Error')}")

            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")