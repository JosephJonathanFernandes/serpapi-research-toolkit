# Narration Script: serpapi-research-toolkit Demo

**Total length:** about 2:30. Read at a calm, steady pace.
**Format:** Read only the quoted lines. *Italic notes* are cues and are not spoken.

### Pronunciation
- **SerpApi**: "serp-A-P-I"
- **LangChain**: "lang-chain"
- **KMeans**: "K-means"
- **TF-IDF**: "T-F-I-D-F"
- **Scholar**: Google Scholar
- **LCEL**: "L-C-E-L"
- **BibTeX**: "bib-tech"

---

## Scene 1: The Problem & Solution (about 30s)
*On screen: the Streamlit app, idle.*

"When you build AI agents or RAG pipelines that search the web, you run into three problems.
First, every search engine returns a different data shape. Google Scholar, web search, news and jobs all use different JSON formats.
Second, agents re-run queries repeatedly, burning through your API credits.
Third, existing tools like langchain-community's SerpAPI wrapper only return raw text or dictionaries—they don't plug into modern LCEL retriever chains.
So we built langchain-serpapi-provider. It's a lightweight, pip-installable LangChain plugin. It normalizes results into one schema, implements BaseRetriever, handles retries, and caches every response on disk."

---

## Scene 1b: The Plugin in Code (about 15s)
*On screen: a code editor showing these two lines, then switch back to the app.*

```python
retriever = SerpApiRetriever(engine="google_scholar", num_results=10)
docs = retriever.invoke("bilingual advantage executive function cognitive control")
```

"This is the whole integration. One retriever, one line to pick the engine, and you get back standard LangChain documents with citations and metadata already attached. You can pipe it straight into an LCEL prompt and model chain."

---

## Scene 2: Live Query & Silhouette Clustering (about 45s)
*On screen: click "Bilingual Advantage", then "Run Research Agent". Keep talking through the loading spinner.*

"Let's try a real research question: does being bilingual improve executive function?
I'll run it now. Three things are happening:
First, our retriever queries Google Scholar through SerpApi, pulling ten papers with authors, citations and links.
Next, instead of returning a flat list, the agent embeds each title and snippet and automatically chooses the best cluster count using silhouette scores.
Finally, it uses T-F-I-D-F phrases to label each cluster, showing the top three most-cited papers alongside so researchers can sanity-check the topic grouping.
And here are the results."

*The cluster cards appear. Move straight into Scene 3.*

---

## Scene 3: Possible Disagreement Detection (about 45s)
*On screen: scroll to the cluster with the amber badge.*

"Look at this cluster. It has an amber badge: 'Possible disagreement — human review recommended.'
We were deliberate with this wording. We do not claim to have built an infallible contradiction detector. Search snippets are truncated and often cut mid-sentence.
Instead, we provide a transparent heuristic filter. If a cluster contains abstracts reporting advantages alongside papers reporting null findings or failed replications, the agent flags it.
Here you can see the cue pills: one paper reports no general cognitive advantages in eleven thousand people, while another discusses advantages. We even built a benchmark in our eval directory showing this transparent heuristic achieves 53% recall and 32% precision, outperforming transformer NLI models that get tripped up by snippet truncation.
The tool doesn't settle scientific disputes. It points researchers directly to papers worth reading closely."

---

## Scene 4: Export, Sorting & Caching (about 25s)
*On screen: point to Export CSV and BibTeX buttons, switch sorting to Citations in the sidebar, then re-click Run.*

"Researchers can export the results directly to CSV or sanitized BibTeX with one click, or re-sort by citations or year.
Now watch what happens when I click Run again on the same query.
It finishes in three seconds—the retrieval itself took under a millisecond from the local disk cache with zero API credits spent.
That means developers can test and iterate on agent workflows without burning their quota."

---

## Scene 5: Close (about 10s)
*On screen: the results page, or the GitHub repo.*

"That's the serpapi research toolkit. A hardened LangChain retriever, normalized cross-engine schemas, built-in caching, and a research assistant that highlights where the literature disagrees. Thanks for watching."

---

### Notes for the recording
- Scene 2 includes the spinner wait, so the narration is paced to fill it. Don't stop talking while it loads.
- Record the screen first, then the voice, and sync them in an editor.
- The repository also includes a one-click 'Demo mode' in the sidebar so anyone can run the app offline without an API key.
