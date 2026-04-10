<div align="center">

[![Website](https://img.shields.io/badge/Website-orgmatcher.vercel.app-00853E?style=flat-square)](https://orgmatcher.vercel.app)
[![Frontend Status](https://img.shields.io/website?url=https%3A%2F%2Forgmatcher.vercel.app&label=Frontend&style=flat-square)](https://orgmatcher.vercel.app)
[![Backend Status](https://img.shields.io/website?url=https%3A%2F%2Forgmatcher-353106949537.us-south1.run.app%2Fdocs&label=Backend&style=flat-square)](https://orgmatcher-353106949537.us-south1.run.app/docs)

</div>

# OrgMatcher

A recommendation engine designed to help students find organizations and clubs at the University of North Texas (UNT) using TF-IDF and other natural language processing techniques.  
Try it out here at **[orgmatcher.vercel.app](https://orgmatcher.vercel.app)**!  

## Documentation 
For a detailed explanation of the goal, implementation, methodology, and process of the project, refer to the project paper at:
**[OrgMatcher Paper](docs/OrgMatcher_Paper.pdf)**.

## Usage
The User will be directly guided to the front page of the website, where they will be prompted to enter anything they are looking for in a campus organization, such as interests, hobbies, and career aspirations.   

<p align="center">
  <img src="docs/images/UI.png" width="80%" alt="UI" />
</p>

Clicking "Find My Orgs" will return the five most relevant organizations that match the user's interests.  
In this example, the user inputs "I am a student interested in Computer Science and looking to make new friends on campus"

<p align="center">
  <img src="docs/images/results.png" width="70%" alt="results" />
</p>

The user can also click "Visit Organization" to go to the organization's webpage on UNT's OrgSync to learn more about the organization.  

<p align="center">
  <img src="docs/images/OrgMatcher_Demo.gif" width="80%" alt="gif" />
</p>

## Tech Stack
- Frontend: React + Vite (Plain JSX)
- Backend: FastAPI + Python
- NLP & Data: Scikit-Learn (TF-IDF, Cosine Similarity), NLTK (Lemmatization), SQLite (database), BeautifulSoup (scraping)
- Deployment: Vercel (frontend), GCP (backend)