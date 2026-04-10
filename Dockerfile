FROM python:3.11-slim

# Set working directory to project root
WORKDIR /app

# Install dependencies first (for better caching)
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Pre-download NLTK data to prevent cold-start delays
RUN python -m nltk.downloader wordnet

# Copy the application and database
COPY backend/ ./backend/
COPY db_and_csvs/ ./db_and_csvs/

# Change working directory to backend where main.py is
WORKDIR /app/backend

# Expose port (Cloud Run sets the PORT env var)
ENV PORT=8000
EXPOSE $PORT

# Run the FastAPI server
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
