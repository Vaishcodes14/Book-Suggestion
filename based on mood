import pandas as pd

# Step 1: Download dataset (you can replace with direct CSV link if you already have it)
# Example link from OpenBigData (Goodreads dataset)
url = "https://github.com/zygmuntz/goodbooks-10k/raw/master/books.csv"

# Step 2: Load the dataset
df = pd.read_csv(url)

print("Original Columns:", df.columns)
print("Total Books in raw dataset:", len(df))

# Step 3: Keep only relevant columns
# Note: Adjust column names if needed (some datasets may have slightly different ones)
columns_to_keep = ["book_id", "title", "authors", "original_publication_year", 
                   "average_rating", "ratings_count", "small_image_url"]

books_df = df[columns_to_keep]

# Step 4: Add placeholder "Summary" (since original goodbooks-10k doesn't have descriptions)
# You can later enrich this with scraped/other metadata
books_df["Summary"] = "No description available yet."

# Step 5: Clean data (drop duplicates & nulls)
books_df = books_df.dropna().drop_duplicates(subset="title")

# Step 6: Save cleaned dataset
books_df.to_csv("books_dataset.csv", index=False, encoding="utf-8")

print(f"✅ Clean dataset saved as 'books_dataset.csv' with {len(books_df)} books.")
