# Legal_Doc_Sim
**Legal Document Similarity using some traditional NLP methods.**

**This work is as direct implementation of the paper:**
**Legal document similarity: a multi-criteria decision-making perspective**
paper available at: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7924540/

## Data set

**Document corpus of court case judgments**: This dataset contains over 9000 
case judgments passed by the Indian Supreme Court. 
Data Available at: 
https://figshare.com/articles/dataset/Document_corpus_of_court_case_judgments/806
3186

## Python packages used

  Gensim
  NLTK
  spaCy
  networkX
  Scikit-learn
  numpy
  pandas
  
## Data pre-processing

Basic lexical processing is used namely: Stop-word removal, punctuation removal, 
and also selective removal of most frequent terms used in almost every legal case 
judgment such as: ['petitioner','petition','respondent','court','appellant','appeal',’trial’] 
etc.
We remove these terms too since they do not contribute towards the main concept of 
the document.
The implemenatation function for this can be found in the file: process.ipynb
This data set contains around 9300 case judgments passed by the Indian Supreme Court. The documents also contain special punctuation such as ‘\n’ to identify next sentence. This makes the task a bit simple for us. In my pre-processing step, I do 3 operations:**remove_stop_words(), remove_punctuation() and remove_legal_words()**.  The remove_punctuation step also removes the ‘\n’ along with all other punctuation. This can be troublesome for us as we have two steps(Tf-IDF and Co-occurrence) in the fine-tuning process where a sentence is treated as an unit for processing. So for this I create 2 different processed document for a single master document. First one is processed with only the **remove_stop_words and remove_legal_words**. Which means, this would contain punctuation and this is passed to the 2 functions **find_tfidf() and co-occurrences()**. Second one is processed with all the three pre-processing steps and this is passed on to functions as find_noun() and also used for our word2vec.


## Fine-tuning -- Main idea of the paper

Since we have to find similarity between two documents, we have to identify the main concept or the main context of the documents, i.e what’s the story behind the judgment. But we do not need any summarization since the concept behind the documents reside in the “concept-words”. We need to identify the concept-words for each of the documents being compared and then check their similarity. We take a rather traditional way of finding semantic meanings from the documents than the more popular Deep Learning based approach.

**What are Concept-Words?**
They are basically the most important nouns that form the story of the document. They form our basic keywords for extraction from the document. However we do not include common nouns in our concept words. If we consider extracting keywords from a news report then a person’s name has very high significance as a concept word since he may be the part of the story and it is more significant because a news report is more often built around stories of people so a common noun there has a relevance. But in our scenario of a legal case judgment we need to understand the case and the judgment passed. A person’s name in a legal case represents a party(respondent or appellant) or a participant in the case and does not actually contribute to any legal concept present in the judgment. So we ignore reference to any specific people or place or any Organization.

**Finding Concept-Words**

After we have preprocessed our text, we apply POS-tagging and identify the nouns in the text. But this won’t be all since we also need to take in account the importance of the word in the document. For finding the importance of the words in the whole document we calculate TF-IDF values of each word from the preprocessed text. Since we need to calculate tf-idf in a single text file or document, here the inverse document frequency of the word is calculated by treating each sentence as a document.

After calculating the tf-idf values of all the words in a document, we calculate the mean of the values. We then take only those words whose tf-idf value is greater than the mean-tfidf. This is justified as this would give those words which almost lies in the center of the Ziph’s distribution of the word ranks and their frequency. This is all we needed since we know that the extremes of the Ziph’s distribution contains words which are unlikely to convey much meaning about the document’s semantics. These words are our [important_words]. Then we find the intersection of the nouns and the important_words. This gives us our concept_words.  
  &emsp;    **Finding nouns:** We use the spaCy python library for finding nouns of the document. In the **find_noun()** function in the **main.ipynb** file, we pass the pre-processed text of a document and this function returns us all the nouns of the document. spaCy internally uses pos_tagging  for every token in the document. The pos_ attribute of every token contains its pos tag.  
   &emsp;    **Important words:** The **find_tfidf()** function takes in the whole pre-processed document and calculates the tf-idf scores for all the words. This function returns us the important_words of the document.  
   &emsp;    **Concept words:** The **find_concept_words()** function takes the nouns and the important words returned from the above mentioned functions and find the intersection between them. The intersection thus contains nouns which are important. These form the concept words of the document.  
      
**Finding Important Concepts** 

A group of concept words that co-occur in a context can form important concepts of the document. From the concept of Distributional Semantics, we know that words that co-occur together in a context, deliver the same meaning. Here we find co-occurrence of the concept-words and form a graph. The graph has vertices as the concept-words and two words have edges if they have co-occurrence count greater than or equal to 3. We consider that if two words co-occur twice then that can be a coincidence or more of a linguistic pattern. The weight of the edges is the co-occurrence count between the the two connecting nodes.We apply Girvan-Newman algorithm to find communities in the graph.  

  &emsp;  **Co-occurrence:** From the concept of Distributional Semantics, we know that words that co-occur together in a context, deliver the same meaning. Here we find co-occurrence of the concept-words and form a graph. Our **co-occurrence()** function actually computes co-occurrence of words in a sentence and stores it in a dictionary. Each entry in the dictionary would look like this:  
{‘section’: { {‘act’ : 2} , {‘form’ : 5}, {‘article’: 15}} }
So in the above example the word ‘section’ co-occurs with ‘act’ in 2 sentences, with ‘form’ in 5 sentences and with ‘article’ in 15 sentences.
When we form a graph with the concept_words, we use the co-occurrence number 15 as the edge weight between the words ‘section’ and ‘article’.  

 &emsp;   **Forming the graph:** We form a graph with nodes as the concept words that we had derived earlier. With the help of the co-occurrence dictionary we form edges only if the co-occurrence number is greater than or equal to 3. This is a hyper-parameter in our model. If we decrease this the number, our graph becomes dense and if we increase the number, it becomes sparse. But this would actually determine how good our communities are. We can tweak this to check with what number we get good communities. We form the graph using the python package **networkX.**  
 
 &emsp; **Community Finding:** We use the Girvan-Newman algorithm to find the communities in the graph.  
 
 **Girvan-Newman algorithm:**
 
 The Girvan-Newman community detection algorithm is a method for detecting communities or clusters in a network or graph. The algorithm works by iteratively removing edges from the graph, with the goal of identifying communities as groups of nodes that are more tightly connected to each other than to the rest of the network.Specifically, the algorithm starts by calculating the betweenness centrality of all edges in the network, which is a measure of how often an edge lies on the shortest path between pairs of nodes. It then removes the edge with the highest betweenness centrality, and recalculates the betweenness centrality of the remaining edges. This process is repeated until all edges have been removed.  
 
 As edges are removed, the network is divided into smaller and smaller components. These components represent communities or clusters of nodes that are more strongly connected to each other than to the rest of the network. The algorithm can be run multiple times, with different starting points, to identify different communities in the network.  
 
 The algorithm will return us with all the communities that are formed. This means that isolated nodes will also be returned as a community if any. For this we apply another hyper-parameter to get good communities. For this implementation we only consider the communities which have more than 3 words in them.  
 
 **NOTE: In the paper they have implemented Louvian Modularity. I have chosen Girvan-Newman as it gave me better results in many documents.**
 
 ## Embedding
 
 We implement a tf-idf weighted word2vec embedding model. This is in some ways better than the normal word2vec embedding because the vectors are scaled according to their tfidf scores.We use the Word2Vec function of the Gensim package for vectorizing the words in the corpus. We use a vector size of 200 and a window  size  of  5  for  this implementation. We pass all the preprocessed documents to the function and store the vectors for each word in a pandas dataframe. The words of the corpus forms the index of the dataframe.
 
 
 ## Cosine-Similarity
 
 We vectorize each of the concepts from the document. We just do a linear addition of the vectors of all the words in a concept and multiply it by (1/n). ‘n’ being the number of words in the concept. So ultimately we get as many vectors as the number of concepts in a particular document.  

Now, for different documents we would have different numbers of concepts per document. Let’s say that we have Doc1 with ‘m’ concepts and Doc2 with ‘n’ concepts, we will be calculating cosine similarity between all the m and n pairs. So we will be calculating mxn similarities in total which we can get in a mxn matrix form.  


## Testing our model: 

Since none of our documents in the data set are labeled, it is hard to make any Accuracy metric for the model. What we do is, we identify 3 documents among which 2 have similar contents and the other one is completely different. If we can show that the known similar documents are having much higher cosine-similarity than similarity of other 2 pairs of document, then we can assert that our model is working as per our expectation.  
      
   I have identified 3 such documents from our data set.  

Document-1:

Concepts:  
[1] {'son', 'mother', 'access', 'paternity', 'plaintiff', 'time', 'begotten', 'law', 'marriage'}  

[2] {'favour', 'wife', 'hindu', 'deed', 'father', 'gift', 'question', 'purposes', 'case', 'husband'}  

[3] {'property', 'share', 'half', 'daughter', 'gifts', 'properties', 'family'}  

As we can see from the concepts, this case is about some marriage dispute.  

Document-2:  

Concepts:  
[1] {'exemption', 'completion', 'tenant', 'question', 'building', 'years', 'period', 'date', 'construction', 'possession'}  


[2] {'rent', 'sub', 'control', 'suit', 'act', 'institution', 'judge', 'section', 'eviction', 'restriction', 'grounds', 'operation'}  

We can guess from the concept words that this case is about some property dispute.  

Document-3:  

Concepts:  
[1] {'thereunder', 'accordance', 'particulars', 'section', 'suits', 'date', 'sub', 'rules', 'agra', 'compliance', 'commencement', 'use', 'force', 'amendment', 'furnish', 'proceedings', 'evidence', 'act', 'provisions', 'clause', 'proviso'}  
[2] {'rupees', 'land', 'rate', 'sir', 'rights', 'suit', 'board', 'tenant', 'holder', 'air', 'ejectment', 'acres', 'revenue', 'khudkasht', 'holders', 'provinces'}  

This document is about some land and property dispute.  


**So according to us, our model should show more similarity between Document-2 and Document-3 but low similarity between Document-1 and Document-3.**

**Similarity between 1&3:**  

Cosine similarity matrix:  
[0.03103316765782009, 0.2327694498844979]  
[0.06263595493154799, 0.2305699154604502]  
[-0.0644272020500727, 0.4463187990221134]  


We can see that the similarity values between all the concepts is very low. The lowest being 0.003 and the highest being 0.44. This says that the 2 documents are very different from each other which we were expecting.  

**Similarity between 2&3:**  

Cosine similarity matrix:  
[0.2978163501814514, 0.4977068164842984]  
[0.69065580204434,  0.4681673320618851]  

As we can see that the similarity values are far more between these 2 documents. The lowest being 0.29 and highest being 0.69. This was what we were expecting as both the documents have more or less the same concept of property dispute.  

**So, finally we can conclude that our model is working as expected**


## Code files:

There are 4 python notebooks.  

**main.ipynb**: this is the main file where all the fine-tuning steps, vectorizing and similarity is calculated.  
**process.ipynb**: this notebook has the pre-processing steps where I process all the documents in the data set and create a csv file to use it later for fine-tuning and word2vec.  
**finetune.ipynb**: this notebook contains demo fine-tuning of one of the documents which contains output of each step.  
**word2Vec.ipynb**: this document implements word2vec for the words in the corpus and creates a csv file to be used during vectorizing.  
