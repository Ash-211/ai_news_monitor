import newspaper

url = "https://speakeasyofstrength.com/random-ass-thoughts/"
article = newspaper.Article(url)
article.download()
article.parse()
article.nlp()

print("Title:", article.title)
print("Keywords:", article.keywords)
print("Summary:", article.summary)
print("Meta keywords:", article.meta_keywords)
print("Meta description:", article.meta_description)
