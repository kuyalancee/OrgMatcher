import OrgCard from './OrgCard'
import './ResultsGrid.css'

function SkeletonCard({ index }) {
  return (
    <div
      className="skeleton-card"
      aria-hidden="true"
      style={{ animationDelay: `${index * 0.15}s` }}
    >
      <div className="skeleton-card__image" />
      <div className="skeleton-card__body">
        <div className="skeleton-card__line skeleton-card__line--title" />
        <div className="skeleton-card__line skeleton-card__line--sub" />
        <div className="skeleton-card__line" />
        <div className="skeleton-card__line" />
        <div className="skeleton-card__line skeleton-card__line--short" />
      </div>
    </div>
  )
}

function ResultsGrid({ results, isLoading }) {
  const showSkeleton = isLoading && results.length === 0
  const showResults = !isLoading && results.length > 0

  if (!showSkeleton && !showResults) return null

  return (
    <section className="results-grid">
      <h2 className="results-grid__heading">Your Top Matches</h2>
      <div className="results-grid__grid">
        {showSkeleton
          ? Array.from({ length: 5 }).map((_, i) => (
              <SkeletonCard key={i} index={i} />
            ))
          : results.map((org, i) => (
              <OrgCard
                key={org.name}
                {...org}
                animationDelay={i * 80}
              />
            ))
        }
      </div>
    </section>
  )
}

export default ResultsGrid
