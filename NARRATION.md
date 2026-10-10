# Narration Script: serpapi-research-toolkit Demo

**Total length:** about 2:30. Read at a calm, steady pace.
**Format:** Read only the quoted lines. *Italic notes* are cues and are not spoken.

### Pronunciation
- **SerpApi**: "serp-A-P-I"
- **LangChain**: "lang-chain"
- **KMeans**: "K-means"
- **TF-IDF**: "T-F-I-D-F"
- **Scholar**: Google Scholar

---

## Scene 1: The Problem (about 25s)
*On screen: the Streamlit app, idle.*

"When you build AI agents that search the web, you run into two problems.
First, every search engine returns a different data shape. Google Scholar, web search, news and jobs all use different formats.
Second, agents re-run the same queries again and again while you develop them, and that burns through your API credits.
So we built langchain-serpapi-provider. It's a lightweight, pip-installable LangChain plugin. It normalizes results into one schema, picks the right search engine automatically, and caches every response on disk."

---

## Scene 1b: The Plugin in Code (about 15s)
*On screen: a code editor showing these two lines, then switch back to the app.*

```python
retriever = SerpApiRetriever(engine="google_scholar", num_results=10)
docs = retriever.invoke("bilingual advantage executive function cognitive control")
```

"This is the whole integration. One retriever, one line to pick the engine, and you get back standard LangChain documents with citations and metadata already attached. The demo you're about to see is built entirely on this."

---

## Scene 2: Live Query (about 50s)
*On screen: click "Bilingual Advantage", then "Run Research Agent". A spinner runs for about 12 seconds. Keep talking through it.*

"Let's try a real research question: does being bilingual improve executive function?
I'll run it now, and this is a live call. Three things are happening.
First, our retriever is querying Google Scholar through SerpApi. It pulls ten papers with author details, citation counts and links.
Next, instead of returning a flat list, the agent embeds each title and abstract and groups related papers with K-means clustering.
Finally, it uses T-F-I-D-F phrases to label each cluster, so you can see at a glance what each group is about.
And here are the results."

*The cluster cards appear. Move straight into Scene 3.*

---

## Scene 3: Mixed Findings Signal (about 45s)
*On screen: scroll to Cluster 2, "Executive and Bilingual", and point at the amber badge.*

"Look at Cluster 2. It has an amber badge: 'Mixed findings signal, human review recommended.'
We were careful with this wording. We are not claiming to detect contradictions with machine learning. It's a simple, transparent heuristic. If a cluster has papers reporting an advantage and papers reporting no effect, it gets flagged.
Here you can see how it works. One paper is titled 'Bilingualism affords no general cognitive advantages', a study of eleven thousand people. Another paper in the same cluster has a snippet mentioning outperforming, which is actually from a definition of the bilingual advantage. The tool only matches phrases, so it can be fooled. It's a pointer, not a verdict.
Whether bilingualism gives an executive-function advantage is widely discussed as contested. The tool doesn't settle that. It only points the researcher to papers worth reading closely."

---

## Scene 4: Caching (about 20s)
*On screen: click "Run Research Agent" a second time. It finishes in about 3 seconds.*

"Now I'll run the same query again.
It finished in about three seconds. The first run took about twelve. The retrieval itself took about a millisecond, because the response came from the local cache and used no API credits.
That means developers can test and iterate on agent workflows without using up their monthly quota."

---

## Scene 5: Close (about 10s)
*On screen: the results page, or the GitHub repo.*

"That's the serpapi research toolkit. A clean LangChain plugin, one normalized schema, built-in caching, and a research assistant that shows you where the literature disagrees. Thanks for watching."

---

### Notes for the recording
- Scene 2 includes the 12-second spinner wait, so the narration is paced to fill it. Don't stop talking while it loads.
- Record the screen first, then the voice, and sync them in an editor.
- Keep the pitch modest. The mixed findings badge is a phrase-based heuristic and has only been tested on a small set of queries.
