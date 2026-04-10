import { useState, useRef } from 'react'
import './SearchBar.css'

function SearchBar({ onSubmit, isLoading }) {
  const [value, setValue] = useState('')
  const [shake, setShake] = useState(false)
  const textareaRef = useRef(null)

  function handleSubmit(e) {
    e.preventDefault()
    const trimmed = value.trim()
    if (!trimmed) {
      setShake(true)
      textareaRef.current?.focus()
      return
    }
    onSubmit(trimmed)
  }

  function handleAnimationEnd() {
    setShake(false)
  }

  return (
    <form className="search-bar" onSubmit={handleSubmit}>
      <textarea
        ref={textareaRef}
        className={`search-bar__textarea${shake ? ' search-bar__textarea--shake' : ''}`}
        rows={4}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onAnimationEnd={handleAnimationEnd}
        placeholder="I enjoy outdoor activities, want to meet people who share my passion for sustainability, and hope to build leadership skills..."
        aria-label="Describe what you're looking for in an organization"
        disabled={isLoading}
      />
      <div className="search-bar__footer">
        <button
          type="submit"
          className="search-bar__button"
          disabled={isLoading}
        >
          {isLoading ? 'Matching\u2026' : 'Find My Orgs \u2192'}
        </button>
      </div>
    </form>
  )
}

export default SearchBar
