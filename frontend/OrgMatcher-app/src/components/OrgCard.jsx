import './OrgCard.css'

function OrgCard({ name, acronym, summary, image_url, org_url, rank, animationDelay }) {
  return (
    <article className="org-card" style={{ animationDelay: `${animationDelay}ms` }}>
      <div className="org-card__image-wrap">
        {image_url ? (
          <img
            className="org-card__image"
            src={image_url}
            alt={name}
          />
        ) : (
          <div className="org-card__image-placeholder" aria-hidden="true">
            🏛️
          </div>
        )}
        <span className="org-card__rank">#{rank}</span>
      </div>

      <div className="org-card__body">
        <h2 className="org-card__name">{name}</h2>
        {acronym && <p className="org-card__acronym">{acronym}</p>}
        <p className="org-card__summary">{summary}</p>
        <a
          className="org-card__link"
          href={org_url}
          target="_blank"
          rel="noopener noreferrer"
        >
          Visit Organization &rarr;
        </a>
      </div>
    </article>
  )
}

export default OrgCard
