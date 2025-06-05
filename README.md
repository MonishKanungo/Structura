*Industrial Grade Ai Solution Structura AI*

Table Extraction AI Suite is an end-to-end, AI-powered platform for extracting, understanding, and querying tabular data from images and PDFs. Leveraging state-of-the-art models like Gemini Vision, LlamaParse, and advanced intent classification, this project transforms raw document images into structured, queryable data—accessible through a modern, interactive React frontend.

**Key Features**
Intelligent Table Extraction: Converts images/PDFs into structured tables using multimodal AI (Gemini Vision, LlamaParse).

Natural Language Q&A: Ask questions in plain English about your extracted tables. Get instant, AI-generated answers and preview relevant data.

Smart Table Intent Classifier: Finds the most relevant table for your question using sentence-transformer embeddings.

Seamless Frontend Experience: Drag-and-drop uploads, real-time status, AI chat, and table previews—all in a beautiful React+Tailwind UI.

Developer-Friendly API: FastAPI backend with clear endpoints for upload, status, and Q&A.

**Real-World Problems Solved**
Engineering BOMs: Automatically extracts and standardizes dimensions and weights (e.g., mm ↔ inch, kg ↔ lb) from scanned BOM tables to prevent part mismatches in manufacturing.

Pharma Formulation Sheets: Converts complex dosage and formulation tables (e.g., mg/ml, °C ↔ °F) from lab notebooks into structured, queryable formats—reducing manual errors.

Oil & Gas Spec Sheets: Translates pressure and flow rate tables (e.g., bar ↔ psi, m³/h ↔ GPM) from equipment manuals for accurate selection and compliance.

**How It Works**
Upload: Users upload images or PDFs via the React app.

Extraction: The backend uses Gemini Vision and LlamaParse to extract tables, headers, and metadata.

Classification: Each table is embedded and indexed for semantic search using a transformer model.

Ask Questions: Users type questions (e.g., “What is the total revenue in 2022?”). The backend finds the best-matching table, generates a context-rich prompt, and uses an LLM to answer.

Preview & Download: See extracted tables and answers instantly. Download results for further analysis.

**Inspiration**
This project is inspired by the need for seamless, AI-driven data extraction and analytics from complex, real-world documents—empowering everyone to unlock insights from their data.

If you’d like any changes or want to emphasize specific features, let me know!

