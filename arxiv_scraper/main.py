

import argparse
from .scraper import get_recent_articles
from .profiler import get_user_profile, get_top_categories
from .utils import calculate_similarity
from datetime import datetime

# A dictionary of arXiv categories and their descriptions
CATEGORIES = {
    'cs.AI': 'Artificial Intelligence',
    'cs.CL': 'Computation and Language',
    'cs.CC': 'Computational Complexity',
    'cs.CE': 'Computational Engineering, Finance, and Science',
    'cs.CG': 'Computational Geometry',
    'cs.GT': 'Computer Science and Game Theory',
    'cs.CV': 'Computer Vision and Pattern Recognition',
    'cs.CY': 'Computers and Society',
    'cs.CR': 'Cryptography and Security',
    'cs.DS': 'Data Structures and Algorithms',
    'cs.DB': 'Databases',
    'cs.DL': 'Digital Libraries',
    'cs.DM': 'Discrete Mathematics',
    'cs.DC': 'Distributed, Parallel, and Cluster Computing',
    'cs.ET': 'Emerging Technologies',
    'cs.FL': 'Formal Languages and Automata Theory',
    'cs.FW': 'Foundations of Computer Science',
    'cs.GL': 'General Literature',
    'cs.GR': 'Graphics',
    'cs.AR': 'Hardware Architecture',
    'cs.HC': 'Human-Computer Interaction',
    'cs.IR': 'Information Retrieval',
    'cs.IT': 'Information Theory',
    'cs.LO': 'Logic in Computer Science',
    'cs.LG': 'Machine Learning',
    'cs.MS': 'Mathematical Software',
    'cs.MA': 'Multiagent Systems',
    'cs.MM': 'Multimedia',
    'cs.NI': 'Networking and Internet Architecture',
    'cs.NE': 'Neural and Evolutionary Computing',
    'cs.NA': 'Numerical Analysis',
    'cs.OS': 'Operating Systems',
    'cs.PF': 'Performance',
    'cs.PL': 'Programming Languages',
    'cs.RO': 'Robotics',
    'cs.SI': 'Social and Information Networks',
    'cs.SE': 'Software Engineering',
    'cs.SD': 'Sound',
    'cs.SC': 'Symbolic Computation',
    'cs.SY': 'Systems and Control',
}

def main():
    parser = argparse.ArgumentParser(description='Scrape arXiv for recent articles and compare them to a user profile.')
    parser.add_argument('profile', type=str, help='Path to the user profile markdown file.')
    args = parser.parse_args()

    user_profile = get_user_profile(args.profile)
    top_categories = get_top_categories(user_profile, CATEGORIES)

    print(f"Top 3 categories for you: {top_categories}")

    articles = []
    for category in top_categories:
        print(f"Scraping {category}...")
        articles.extend(get_recent_articles(category))

    for article in articles:
        # Combine title and abstract for similarity matching
        text_to_match = article['title'] + ' ' + article['abstract']
        similarity = calculate_similarity(user_profile, text_to_match)
        article['similarity'] = similarity

    sorted_articles = sorted(articles, key=lambda x: x['similarity'], reverse=True)

    # Generate filename with current date
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"arxiv_digest_{date_str}.md"

    output_content = f"# arXiv Digest for {date_str}\n\n"
    output_content += f"Top 3 categories for you: {top_categories}\n\n"
    for article in sorted_articles:
        output_content += f"## {article['title']}\n\n"
        output_content += f"**Link:** [{article['link']}]({article['link']})\n\n"
        output_content += f"**Similarity:** {article['similarity']:.4f}\n\n"
        output_content += f"### Abstract\n\n"
        output_content += f"{article['abstract']}\n\n"
        output_content += "---\n\n"

    with open(filename, 'w') as f:
        f.write(output_content)

    print(f"Output saved to {filename}")

if __name__ == '__main__':
    main()
