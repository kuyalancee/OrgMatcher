import { useState } from 'react'
import './App.css'
import SearchBar from './components/SearchBar'
import ResultsGrid from './components/ResultsGrid'

function App() {
  const [results, setResults] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSearch(query) {
    setIsLoading(true)
    setError(null)
    try {
      const apiUrl = import.meta.env.VITE_API_URL || '';
      const res = await fetch(`${apiUrl}/api/match`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      })
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const data = await res.json()
      setResults(data.results)
    } catch (err) {
      setError('Something went wrong. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app__header">
        <h1 className="app__title">
          OrgMatcher<span className="app__title-dot">.</span>
        </h1>
        <p className="app__tagline">Find your people at UNT.</p>
      </header>

      <div className="app__search-section">
        <SearchBar onSubmit={handleSearch} isLoading={isLoading} />
      </div>

      {error && (
        <p className="app__error">{error}</p>
      )}

      <div className="app__results-section">
        <ResultsGrid results={results} isLoading={isLoading} />
      </div>
    </div>
  )
}

export default App
