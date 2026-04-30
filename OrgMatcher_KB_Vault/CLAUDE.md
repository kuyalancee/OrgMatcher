# Project : OrgMatcher

OrgMatcher is a Natural Language Processing (NLP) program designed to match students with suitable campus organizations. It prompts the user for a string of text outlining the qualities they look for in an organization, such as hobbies, interests, social connections, personal growth, and other personal factors. The program uses scraped data from over 400 UNT organizations, including their name, acronym, summary, and description. By matching the terms in the user’s input against the content of each organization’s contents from the UNT OrgSync page, the program returns the five most relevant organizations based on compatibility using NLP methods such as TF-IDF, lemmatization, and cosine similarity.

## Architecture: 
frontend/ : React and Vite frameworks
backend/ : FastAPI framework

## Rules: 
- Never delete files or directories without informing the user first
- Before writing to any file, verify it aligns with the prompt and project structure
- Before architectural changes, check OrgMatcher_KB_Vault/ for relevant docs — but only read
  files pertinent to the current task, not all of them
- After finalizing decisions, fixing major bugs, or explaining complex topics, write a summary
  to OrgMatcher_KB_Vault/scribe/ as YYYY-MM-DD_Topic-Summary.md including:
  1. Goal/issue overview
  2. Key decisions or architectural changes
  3. Files modified
- Manual trigger: "wrap up", "save session", or "summarize this" → immediately write summary
