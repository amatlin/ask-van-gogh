---
layout: post
title:  "Studying Van Gogh with Retrieval Augmented Generation (RAG)"
date:   2024-01-21 22:14:58 -0800
categories: jekyll update
---

## Introduction 

The [Van Gogh letters](https://www.vangoghmuseum.nl/en/art-and-stories/stories/all-stories/van-goghs-letters) offer a view into the painter's innermost thoughts, sources of artistic inspiration, and relationships. More than 100 years since the publication of the letters by his sister-in-law, Johanna van Gogh-Bonger, Van Gogh's writings can be understood in new ways thanks to the introduction of nifty LLM techniques. In particular, Retrieval Augmented Generation (RAG) creates an opportunity to "talk to" a corpus of documents. This blog post is a learning project in which I create a RAG pipeline based on the Van Gogh letters, demonstrating how such an approach enhances the performance of large language models on domain-specific questions. 

More concretely, we will explore how to get from regular ChatGPT...

```
Original question
Standard ChatGPT answer
```

... to our enhanced version of ChatGPT, which we will call *AskVanGogh*. 

```
Same question
Fancy answer
```

Interestingly, a similar project was done by the startup working with the Musée d'Orsay in Paris. 

## Methodology

In the sections below, we will build out AskVanGogh, step by step. If you are interested to download the code and try it out for yourself, please check out the [ask-van-gogh](https://github.com/amatlin/ask-van-gogh) Github repository. 

- Details on the dataset (Van Gogh's letters)
- Process of integrating Van Gogh's letters with RAG.
  - Split letters into chunks based on a maximum number of tokens
  - Embed letter chunks and inspect patterns
  - Prompt engineering
- Technical overview of the RAG model.
- Challenges faced during the project.

### The Van Gogh Letters
Notebook link: `parse_letters.ipynb`

- didn't have a convenient database of letters at my disposal
- had to find a website that would have the letters available in an easily consumable format
- found this website, which has most of the letters available as PDFs
- there is also handy metadata which tells us the location, sender, and recipient of the letter
- noticed that the letters are all in english, which means that they are translated already
- nice trick to write beautifulsoup scripts with chatgpt
- future improvements: use an opensource library such as llamaindex to load PDF documents
- because our data is small, we don't have to worry about storage constraints, and can just write to csv
- resulting dataframe head
- some descriptive statistics
  - letters grouped by sender, recipient
  - letters grouped by origin

### Creating an index 
Notebook links: `embed_letters.ipynb`

In order for large language models to "read" the letters, we must first encode the natural language into tokens using the `tiktoken` library from OpenAI. These tokens belong to a large but finite vocabulary that is familiar to the model. We scan through the letters and break them into chunks if the number of tokens is greater than the model's context window (source: [OpenAI Cookbook: Embedding Wikipedia articles for search](https://cookbook.openai.com/examples/embedding_wikipedia_articles_for_search)). Then, we feed the letter chunks into the `text-embedding-ada-002` model to obtain high-dimensional embeddings. It is important to note that embeddings from recent transformer-based models have contextual understanding, meaning that a single word may have different embeddings depending on the words surrounding it. 

```
# Split letters into chunks
MAX_TOKENS = 1600
vgogh_letter_chunks = []
idxs = []

for idx, letter in tqdm(letters.iterrows()):
   split_result = split_strings_from_subsection(letter.Text, max_tokens=MAX_TOKENS)
   vgogh_letter_chunks.extend(split_result)
   len_chunks = len(split_result)
   idxs.extend([idx] * len_chunks)

# Join chunks to their letter metadata and drop the full, original text
vgogh_letter_chunks_df = pd.DataFrame({'chunk': vgogh_letter_chunks, 'idx': idxs}).set_index('idx').join(letters).drop('Text', axis=1)
```

```
BATCH_SIZE = 100  

embeddings = []
for batch_start in range(0, len(vgogh_letter_chunks), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch = vgogh_letter_chunks[batch_start:batch_end]
    print(f"Batch {batch_start} to {min(len(vgogh_letter_chunks), batch_end-1)}")
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
    for i, be in enumerate(response.data):
        assert i == be.index  # double check embeddings are in same order as input
    batch_embeddings = [e.embedding for e in response.data]
    embeddings.extend(batch_embeddings)
```

```
import umap 
import plotly.express as px

# Reduce dimensionality of embeddings to 2D with UMAP
umap_model = umap.UMAP(n_components=5, metric='cosine', random_state=42).fit(vgogh_letter_chunks_df.embeddings.tolist())

umap_embeddings = umap_model.transform(vgogh_letter_chunks_df.embeddings.tolist())

# Add the UMAP embeddings to your DataFrame
vgogh_letter_chunks_df['UMAP-1'] = umap_embeddings[:, 0]
vgogh_letter_chunks_df['UMAP-2'] = umap_embeddings[:, 1]

# Create an interactive scatter plot with Plotly Express
fig = px.scatter(vgogh_letter_chunks_df, x='UMAP-1', y='UMAP-2', 
                 color='Origin', facet_col='From', facet_row='To', 
                 title='UMAP projection of Vincent van Gogh Letters',
                 hover_data=['Origin', 'From', 'To'])

# Show the plot
fig.show()
```

To further understand 

- What are embeddings 
- How do we choose a set of embeddings from OpenAI library
- Filter to Van Gogh letters only and perform UMAP visualization
  - inspect the clusters to understand what the different themes are


### Prompt engineering

- Starter prompt
- Telling GPT to behave as van gogh 
- 

### Putting it all together


<!-- 

You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

Jekyll requires blog post files to be named according to the following format:

`YEAR-MONTH-DAY-title.MARKUP`

Where `YEAR` is a four-digit number, `MONTH` and `DAY` are both two-digit numbers, and `MARKUP` is the file extension representing the format used in the file. After that, include the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
-->

## Conclusion
- Impact of RAG on the museum experience