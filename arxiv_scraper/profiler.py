import markdown
from bs4 import BeautifulSoup
from collections import Counter

def get_user_profile(profile_path):
    """
    Reads a user's profile from a markdown file.
    """
    with open(profile_path, 'r') as f:
        md_content = f.read()
    
    html = markdown.markdown(md_content)
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()

def get_top_categories(profile_text, categories):
    """
    Identifies the top 3 categories based on the user's profile.
    """
    profile_text_lower = profile_text.lower()
    category_counts = Counter()
    
    for category, description in categories.items():
        # Count occurrences of category name and description keywords
        count = profile_text_lower.count(category.lower())
        for word in description.split():
            count += profile_text_lower.count(word.lower())
        category_counts[category] = count
        
    return [category for category, count in category_counts.most_common(3)]