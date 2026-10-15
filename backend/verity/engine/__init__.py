"""The retrieval engine: discover files, read them, chunk them, search them.

Nothing in here knows about HTTP, the database, or a language model. It turns
a directory of documents into searchable chunks with provenance, and answers
retrieval queries against them.
"""
